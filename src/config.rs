use crate::{
    graph::Graph,
    store::{GenericGradientMap1, GenericGradientMapN},
};

/// Configuration trait for `Graph`.
pub trait Config {
    /// How to compute forward values.
    type EvalAlgebra: Default + Clone;

    /// How to compute gradient values.
    type GradientAlgebra;

    /// How to store gradients.
    type GradientStore;
}

/// Configuration object for first order differentials.
pub struct Config1<E>(std::marker::PhantomData<E>);

/// Configuration object for higher-order differentials.
pub struct ConfigN<E>(std::marker::PhantomData<E>);

impl<E: Default + Clone> Config for Config1<E> {
    type EvalAlgebra = E;
    type GradientAlgebra = E;
    type GradientStore = GenericGradientMap1;
}

impl<E: Default + Clone> Config for ConfigN<E> {
    type EvalAlgebra = E;
    type GradientAlgebra = Graph<ConfigN<E>>;
    type GradientStore = GenericGradientMapN;
}
