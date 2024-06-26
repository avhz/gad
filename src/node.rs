use crate::config::Config;
use crate::store::Id;
use std::sync::Arc;

type GradientUpdateFunc<C> = Arc<
    dyn Fn(
            &mut <C as Config>::GradientAlgebra, // Algebra to for gradient computation
            &mut <C as Config>::GradientStore,   // Store to keep track of gradients
            Id,                                  // Index of the output gradient
        ) -> Result<(), crate::error::Error>
        + Send
        + Sync,
>;

/// A computational node tracked in the graph.
pub struct Node<C: Config> {
    /// Track dependencies.
    pub(crate) inputs: Vec<Option<Id>>,
    /// Function for updating the gradient of the input variables.
    pub(crate) update_func: Option<GradientUpdateFunc<C>>,
}

impl<C: Config> Node<C> {
    pub(crate) fn clear(&mut self) {
        self.inputs.clear();
        self.update_func = None;
    }
}

impl<C: Config> Clone for Node<C> {
    fn clone(&self) -> Self {
        Self {
            inputs: self.inputs.clone(),
            update_func: self.update_func.clone(),
        }
    }
}

impl<C: Config> std::fmt::Debug for Node<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Node")
            .field("inputs", &self.inputs)
            .finish()
    }
}
