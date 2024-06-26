use crate::store::{GradientId, Id};

/// A value tracked in a graph.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Value<D> {
    /// Forward value.
    pub(crate) data: D,

    /// Handle on the computational node, if any.
    /// * This is also used to index gradients in the gradient store.
    /// * None for constants.
    pub(crate) id: Option<GradientId<D>>,
}

impl<D> Value<D> {
    /// Create a constant valid in any graph-based algebra.
    /// This is safe because constants are not tracked in the graph.
    pub fn constant(data: D) -> Self {
        Value { data, id: None }
    }

    /// The data of a computation node.
    pub fn data(&self) -> &D {
        &self.data
    }

    /// The id of a computation node.
    pub fn id(&self) -> Option<GradientId<D>> {
        self.id
    }

    /// The internal, untyped id of a computation node (used to track dependencies).
    pub fn input(&self) -> Option<Id> {
        self.id.map(|id| id.inner)
    }
}
