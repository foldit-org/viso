//! Dynamic GPU buffer management with automatic resizing
//!
//! Provides buffers that grow automatically when data exceeds capacity,
//! using a 2x growth strategy to minimize reallocations.  Buffers also
//! shrink after sustained underutilization to reclaim VRAM.

use wgpu::util::DeviceExt;

/// Consecutive underutilized writes before shrinking (~1 s at 60 fps).
const SHRINK_AFTER_WRITES: u32 = 60;

/// Minimum capacity below which shrinking is not worth the overhead.
const SHRINK_MIN_CAPACITY: usize = 4096;

/// A GPU buffer that grows and shrinks dynamically.
///
/// Uses a 2× growth strategy when capacity is exceeded.  Shrinks when
/// utilization stays below 25 % of capacity for [`SHRINK_AFTER_WRITES`]
/// consecutive writes, reallocating to 2× the used size.
pub(crate) struct DynamicBuffer {
    buffer: wgpu::Buffer,
    capacity: usize, // Capacity in bytes
    len: usize,      // Current data length in bytes
    usage: wgpu::BufferUsages,
    label: String,
    /// Consecutive writes where `len < capacity / 4`.
    writes_underutilized: u32,
}

impl DynamicBuffer {
    /// Buffer with the given initial byte capacity.
    pub(crate) fn new(
        device: &wgpu::Device,
        label: &str,
        initial_capacity: usize,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let capacity = initial_capacity.max(64); // Minimum 64 bytes

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            capacity,
            len: 0,
            usage,
            label: label.to_owned(),
            writes_underutilized: 0,
        }
    }

    /// Buffer initialized from existing data.
    pub(crate) fn new_with_data<T: bytemuck::Pod>(
        device: &wgpu::Device,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> Self {
        let data_bytes = bytemuck::cast_slice(data);
        let capacity = data_bytes.len().max(64);

        // Allocate the full (floored) capacity so the tracked `capacity`
        // always equals the real buffer size. Without this, a buffer
        // initialized with fewer than 64 bytes would report `capacity` 64
        // while the backing buffer was only `data_bytes.len()` bytes, and a
        // later grow-check (`needed > capacity`) would wrongly skip
        // reallocation and overrun the buffer on write. The padding bytes are
        // zeroed and unused (`len` tracks the real data length). This also
        // turns the empty-data case into a valid 64-byte buffer.
        let mut contents = data_bytes.to_vec();
        contents.resize(capacity, 0);

        let buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &contents,
                usage: usage | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            buffer,
            capacity,
            len: data_bytes.len(),
            usage,
            label: label.to_owned(),
            writes_underutilized: 0,
        }
    }

    /// Write data to buffer, growing if necessary.
    ///
    /// Returns `true` if buffer was reallocated (bind groups need recreation).
    pub(crate) fn write<T: bytemuck::Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[T],
    ) -> bool {
        self.write_bytes(device, queue, bytemuck::cast_slice(data))
    }

    /// Write raw bytes to buffer, growing or shrinking as needed.
    ///
    /// Returns `true` if buffer was reallocated (bind groups need recreation).
    pub(crate) fn write_bytes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
    ) -> bool {
        let needed = data.len();

        // Track sustained underutilization for shrink policy.
        if needed < self.capacity / 4 && self.capacity > SHRINK_MIN_CAPACITY {
            self.writes_underutilized += 1;
        } else {
            self.writes_underutilized = 0;
        }

        let reallocated = if needed > self.capacity {
            // Growth: 2× with a 256 MB hard cap.
            const MAX_BUFFER: usize = 256 * 1024 * 1024;
            let new_capacity = (needed * 2)
                .max(self.capacity + 1024)
                .min(MAX_BUFFER)
                .max(needed);

            self.reallocate(device, new_capacity);
            self.writes_underutilized = 0;
            true
        } else if self.writes_underutilized >= SHRINK_AFTER_WRITES {
            // Shrink: utilization has been <25 % long enough.
            let new_capacity = (needed * 2).max(64);
            log::debug!(
                "Shrinking buffer '{}': {} → {} bytes (used: {})",
                self.label,
                self.capacity,
                new_capacity,
                needed,
            );
            self.reallocate(device, new_capacity);
            self.writes_underutilized = 0;
            true
        } else {
            false
        };

        if needed > 0 {
            queue.write_buffer(&self.buffer, 0, data);
        }
        self.len = needed;

        reallocated
    }

    /// Replace the backing buffer with one of `new_capacity` bytes.
    fn reallocate(&mut self, device: &wgpu::Device, new_capacity: usize) {
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&self.label),
            size: new_capacity as u64,
            usage: self.usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.capacity = new_capacity;
    }

    /// Returns a reference to the underlying `wgpu::Buffer`.
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Returns the current data length in bytes.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Returns the allocated capacity in bytes.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Typed wrapper for DynamicBuffer with cleaner API
///
/// Tracks item count rather than byte length.
pub(crate) struct TypedBuffer<T> {
    inner: DynamicBuffer,
    count: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> TypedBuffer<T> {
    /// Typed buffer initialized from existing data.
    pub(crate) fn new_with_data(
        device: &wgpu::Device,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> Self {
        Self {
            inner: DynamicBuffer::new_with_data(device, label, data, usage),
            count: data.len(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Write data to buffer, growing if necessary
    ///
    /// Returns `true` if buffer was reallocated (bind groups need recreation)
    pub(crate) fn write(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[T],
    ) -> bool {
        self.count = data.len();
        self.inner.write(device, queue, data)
    }

    /// Write raw bytes to buffer, growing if necessary.
    ///
    /// Infers item count from byte length / type size.
    /// Returns `true` if buffer was reallocated (bind groups need recreation).
    pub(crate) fn write_bytes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
    ) -> bool {
        self.count = data.len() / size_of::<T>();
        self.inner.write_bytes(device, queue, data)
    }

    /// Returns a reference to the underlying `wgpu::Buffer`.
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        self.inner.buffer()
    }

    /// Returns the current data length in bytes.
    pub(crate) fn len_bytes(&self) -> usize {
        self.inner.len()
    }

    /// Returns the allocated capacity in bytes.
    pub(crate) fn capacity_bytes(&self) -> usize {
        self.inner.capacity()
    }
}
