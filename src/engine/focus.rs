//! Focus state for tab cycling between entities.

use molex::entity::molecule::id::EntityId;

/// Focus state for tab cycling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Focus {
    /// All entities.
    #[default]
    All,
    /// A specific entity.
    Entity(EntityId),
}
