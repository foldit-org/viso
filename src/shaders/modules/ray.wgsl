// Ray-primitive intersections and surface normals.
//
// Consolidates all analytic ray-vs-geometry tests used by impostor and
// picking shaders: sphere, capsule, cone, and AABB.

#define_import_path viso::ray

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
};

/// Ray-sphere intersection via the quadratic formula.
/// Returns the nearest positive `t`, or a negative value on miss.
fn intersect_sphere(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    radius: f32,
) -> f32 {
    let oc = ray_origin - center;
    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(oc, ray_dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;

    if (disc < 0.0) {
        return -1.0;
    }

    let t = (-b - sqrt(disc)) / (2.0 * a);
    if (t < 0.001) {
        return -1.0;
    }

    return t;
}

/// Ray-capsule intersection (cylinder with hemispherical caps).
/// Returns `vec3(t, axis_param, hit_type)` where:
///   - `t`          = ray parameter (negative means miss)
///   - `axis_param` = 0..1 position along capsule axis
///   - `hit_type`   = 1.0 cylinder, 2.0 cap A, 3.0 cap B
fn intersect_capsule(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    cap_a: vec3<f32>,
    cap_b: vec3<f32>,
    radius: f32
) -> vec3<f32> {
    let ba = cap_b - cap_a;
    let ba_len = length(ba);

    if (ba_len < 0.0001) {
        let oc = ray_origin - cap_a;
        let a = dot(ray_dir, ray_dir);
        let b = 2.0 * dot(oc, ray_dir);
        let c = dot(oc, oc) - radius * radius;
        let disc = b * b - 4.0 * a * c;
        if (disc < 0.0) { return vec3<f32>(-1.0, 0.0, 0.0); }
        let t = (-b - sqrt(disc)) / (2.0 * a);
        if (t > 0.001) { return vec3<f32>(t, 0.5, 2.0); }
        return vec3<f32>(-1.0, 0.0, 0.0);
    }

    let ba_dir = ba / ba_len;
    let oa = ray_origin - cap_a;

    let ray_par = dot(ray_dir, ba_dir);
    let ray_perp = ray_dir - ba_dir * ray_par;
    let oa_par = dot(oa, ba_dir);
    let oa_perp = oa - ba_dir * oa_par;

    let a = dot(ray_perp, ray_perp);
    let b = 2.0 * dot(oa_perp, ray_perp);
    let c = dot(oa_perp, oa_perp) - radius * radius;

    var best_t: f32 = 1e20;
    var best_axis: f32 = 0.0;
    var best_type: f32 = 0.0;

    let disc_cyl = b * b - 4.0 * a * c;
    if (disc_cyl >= 0.0 && a > 0.0001) {
        let sqrt_disc = sqrt(disc_cyl);
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);

        for (var i = 0; i < 2; i++) {
            let t = select(t2, t1, i == 0);
            if (t > 0.001 && t < best_t) {
                let axis_pos = oa_par + ray_par * t;
                if (axis_pos >= 0.0 && axis_pos <= ba_len) {
                    best_t = t;
                    best_axis = axis_pos / ba_len;
                    best_type = 1.0;
                }
            }
        }
    }

    // Check cap A
    {
        let oc = ray_origin - cap_a;
        let a_sph = dot(ray_dir, ray_dir);
        let b_sph = 2.0 * dot(oc, ray_dir);
        let c_sph = dot(oc, oc) - radius * radius;
        let disc = b_sph * b_sph - 4.0 * a_sph * c_sph;
        if (disc >= 0.0) {
            let t = (-b_sph - sqrt(disc)) / (2.0 * a_sph);
            if (t > 0.001 && t < best_t) {
                let hit = ray_origin + ray_dir * t;
                let axis_pos = dot(hit - cap_a, ba_dir);
                if (axis_pos <= 0.0) {
                    best_t = t;
                    best_axis = 0.0;
                    best_type = 2.0;
                }
            }
        }
    }

    // Check cap B
    {
        let oc = ray_origin - cap_b;
        let a_sph = dot(ray_dir, ray_dir);
        let b_sph = 2.0 * dot(oc, ray_dir);
        let c_sph = dot(oc, oc) - radius * radius;
        let disc = b_sph * b_sph - 4.0 * a_sph * c_sph;
        if (disc >= 0.0) {
            let t = (-b_sph - sqrt(disc)) / (2.0 * a_sph);
            if (t > 0.001 && t < best_t) {
                let hit = ray_origin + ray_dir * t;
                let axis_pos = dot(hit - cap_a, ba_dir);
                if (axis_pos >= ba_len) {
                    best_t = t;
                    best_axis = 1.0;
                    best_type = 3.0;
                }
            }
        }
    }

    if (best_type == 0.0) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }

    return vec3<f32>(best_t, best_axis, best_type);
}

/// Surface normal at a point on a capsule.
fn capsule_normal(
    point: vec3<f32>,
    cap_a: vec3<f32>,
    cap_b: vec3<f32>,
    hit_type: f32
) -> vec3<f32> {
    if (hit_type == 2.0) {
        return normalize(point - cap_a);
    } else if (hit_type == 3.0) {
        return normalize(point - cap_b);
    } else {
        let axis = normalize(cap_b - cap_a);
        let to_point = point - cap_a;
        let proj = axis * dot(to_point, axis);
        return normalize(to_point - proj);
    }
}

/// Ray-cone intersection.
/// Returns `vec3(t, u, hit_type)` where:
///   - `t`        = ray parameter (negative means miss)
///   - `u`        = 0..1 position along axis (0 = base, 1 = tip)
///   - `hit_type` = 1.0 cone surface, 2.0 base cap, 0.0 miss
fn intersect_cone(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    base: vec3<f32>,
    tip: vec3<f32>,
    base_radius: f32
) -> vec3<f32> {
    let axis = tip - base;
    let height = length(axis);

    if (height < 0.0001) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }

    let axis_dir = axis / height;

    // Cone equation: at height h along axis, radius = base_radius * (1 - h/height)
    let cos_angle = height / sqrt(height * height + base_radius * base_radius);
    let sin_angle = base_radius / sqrt(height * height + base_radius * base_radius);
    let cos2 = cos_angle * cos_angle;
    let sin2 = sin_angle * sin_angle;

    // Transform ray to cone space (tip at origin, axis along +Y)
    let co = ray_origin - tip;

    let d_dot_v = dot(ray_dir, axis_dir);
    let co_dot_v = dot(co, axis_dir);

    // Quadratic coefficients for infinite cone
    let a = d_dot_v * d_dot_v - cos2;
    let b = 2.0 * (d_dot_v * co_dot_v - dot(ray_dir, co) * cos2);
    let c_coef = co_dot_v * co_dot_v - dot(co, co) * cos2;

    var best_t: f32 = 1e20;
    var best_u: f32 = 0.0;
    var best_type: f32 = 0.0;

    // Check cone surface
    let disc = b * b - 4.0 * a * c_coef;
    if (disc >= 0.0 && abs(a) > 0.0001) {
        let sqrt_disc = sqrt(disc);
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);

        for (var i = 0; i < 2; i++) {
            let t = select(t2, t1, i == 0);
            if (t > 0.001 && t < best_t) {
                let hit = ray_origin + ray_dir * t;
                let hit_along_axis = dot(hit - base, axis_dir);
                // Check if hit is within cone bounds (between base and tip)
                if (hit_along_axis >= 0.0 && hit_along_axis <= height) {
                    best_t = t;
                    best_u = hit_along_axis / height;
                    best_type = 1.0;
                }
            }
        }
    }

    // Check base cap (disc at base)
    let denom = dot(ray_dir, axis_dir);
    if (abs(denom) > 0.0001) {
        let t = dot(base - ray_origin, axis_dir) / denom;
        if (t > 0.001 && t < best_t) {
            let hit = ray_origin + ray_dir * t;
            let dist_from_axis = length(hit - base - axis_dir * dot(hit - base, axis_dir));
            if (dist_from_axis <= base_radius) {
                best_t = t;
                best_u = 0.0;
                best_type = 2.0;
            }
        }
    }

    if (best_type == 0.0) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }

    return vec3<f32>(best_t, best_u, best_type);
}

/// Surface normal at a point on a cone.
fn cone_normal(
    point: vec3<f32>,
    base: vec3<f32>,
    tip: vec3<f32>,
    base_radius: f32,
    hit_type: f32
) -> vec3<f32> {
    let axis = tip - base;
    let height = length(axis);
    let axis_dir = axis / height;

    if (hit_type == 2.0) {
        // Base cap — normal points away from tip
        return -axis_dir;
    } else {
        // Cone surface
        let to_point = point - base;
        let along_axis = dot(to_point, axis_dir);
        let radial = to_point - axis_dir * along_axis;
        let radial_dir = normalize(radial);

        // The normal tilts inward toward the axis as we go up
        // tan(half_angle) = base_radius / height
        let slope = base_radius / height;
        return normalize(radial_dir + axis_dir * slope);
    }
}

/// Ray-AABB intersection via slab method.
/// Returns `(t_near, t_far)`. Negative `t_near` means miss.
fn intersect_aabb(ray: Ray, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / ray.direction;
    let t0 = (box_min - ray.origin) * inv_dir;
    let t1 = (box_max - ray.origin) * inv_dir;
    let t_min = min(t0, t1);
    let t_max = max(t0, t1);
    let t_near = max(t_min.x, max(t_min.y, t_min.z));
    let t_far = min(t_max.x, min(t_max.y, t_max.z));
    if (t_near > t_far || t_far < 0.0) {
        return vec2<f32>(-1.0, -1.0);
    }
    return vec2<f32>(t_near, t_far);
}

/// Ray-AABB entry/exit with `t_near` clamped to 0 when the ray origin is
/// inside the box.
fn ray_box_entry_exit(ray: Ray, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let t = intersect_aabb(ray, box_min, box_max);
    if (t.x < 0.0 && t.y < 0.0) {
        return t;
    }
    return vec2<f32>(max(t.x, 0.0), t.y);
}
