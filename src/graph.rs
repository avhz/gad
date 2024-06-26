// Copyright (c) Facebook, Inc. and its affiliates
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::config::{Config, Config1, ConfigN};
use crate::{
    algebras::core::{CoreAlgebra, HasDims},
    error::{check_equal_dimensions, Error, Result},
    node::Node,
    store::{
        GenericGradientMap1, GenericGradientMapN, GradientId, GradientStore, GraphArenaBehavior, Id,
    },
    value::Value,
};
use std::{collections::BinaryHeap, sync::Arc};

#[cfg(doc)]
use crate::prelude::*;

/// Main structure holding the computational graph (aka "tape") used for automatic differentiation.
/// In practice, the configuration is instantiated to build either [`Graph1`] or [`GraphN`],
/// depending if higher-order differentials are needed or not.
pub struct Graph<C: Config> {
    nodes: id_arena::Arena<Node<C>, GraphArenaBehavior>,
    eval: C::EvalAlgebra,
}

type GradientUpdateFunc<C> = Arc<
    dyn Fn(
            &mut <C as Config>::GradientAlgebra, // Algebra to for gradient computation
            &mut <C as Config>::GradientStore,   // Store to keep track of gradients
            Id,                                  // Index of the output gradient
        ) -> Result<()>
        + Send
        + Sync,
>;

impl<C: Config> Default for Graph<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Config> Graph<C> {
    /// Create a new graph.
    pub fn new() -> Self {
        Self {
            nodes: id_arena::Arena::new(),
            eval: C::EvalAlgebra::default(),
        }
    }

    #[inline]
    pub fn eval(&mut self) -> &mut C::EvalAlgebra {
        &mut self.eval
    }
}

impl<C: Config> Graph<C> {
    #[inline]
    pub(crate) fn make_variable<D>(&mut self, data: D) -> Value<D> {
        let node = Node {
            inputs: Vec::new(),
            update_func: None,
        };
        let id = Some(GradientId::new(self.nodes.alloc(node)));
        Value { id, data }
    }

    /// Create a computation node (used to define operators).
    /// During back-propagation, `update_func` must call `store.add_gradient` to propagate the gradient
    /// of each (non-constant) input.
    pub fn make_node<D, G, F, Dims>(
        &mut self,
        data: D,
        inputs: Vec<Option<Id>>,
        update_func: F,
    ) -> Value<D>
    where
        C::GradientAlgebra: CoreAlgebra<D, Value = G>,
        C::GradientStore: GradientStore<GradientId<D>, G>,
        D: HasDims<Dims = Dims>,
        G: HasDims<Dims = Dims> + Clone + 'static,
        Dims: PartialEq + std::fmt::Debug + Clone + 'static + Send + Sync,
        F: Fn(&mut C::GradientAlgebra, &mut C::GradientStore, G) -> Result<()>
            + 'static
            + Send
            + Sync,
    {
        self.make_generic_node::<D, D, G, G, F, Dims>(data, inputs, update_func)
    }

    /// Create a computation node where the source type `S` may be different than the target type `D`.
    pub fn make_generic_node<S, D, GS, GD, F, Dims>(
        &mut self,
        data: D,
        inputs: Vec<Option<Id>>,
        update_func: F,
    ) -> Value<D>
    where
        C::GradientAlgebra: CoreAlgebra<S, Value = GS>,
        C::GradientAlgebra: CoreAlgebra<D, Value = GD>,
        C::GradientStore: GradientStore<GradientId<D>, GD>,
        C::GradientStore: GradientStore<GradientId<S>, GS>,
        D: HasDims<Dims = Dims>,
        GD: HasDims<Dims = Dims> + Clone + 'static,
        Dims: PartialEq + std::fmt::Debug + Clone + 'static + Send + Sync,
        F: Fn(&mut C::GradientAlgebra, &mut C::GradientStore, GD) -> Result<()>
            + 'static
            + Send
            + Sync,
    {
        if inputs.iter().all(|id| id.is_none()) {
            return Value::constant(data);
        }
        let dims = data.dims();
        let update_func: GradientUpdateFunc<C> =
            Arc::new(move |algebra, store, index| -> Result<()> {
                let value: GD = store
                    .get(GradientId::<D>::new(index))
                    .ok_or_else(|| Error::missing_gradient(func_name!()))?
                    .clone();
                check_equal_dimensions(func_name!(), &[&value.dims(), &dims])?;
                update_func(algebra, store, value)
            });
        let node = Node {
            inputs,
            update_func: Some(update_func),
        };
        let id = Some(GradientId::new(self.nodes.alloc(node)));
        Value { id, data }
    }
}

/// Core implementation of the automatic differentiation.
/// We derive more precise variants below to facilitate type inference.
impl<C: Config> Graph<C> {
    #[inline]
    fn do_compute_gradients<D, G>(
        &self,
        graph: &mut C::GradientAlgebra,
        gid: GradientId<D>,
        gradient: G,
    ) -> Result<C::GradientStore>
    where
        C::GradientAlgebra: CoreAlgebra<D, Value = G>,
        C::GradientStore: GradientStore<GradientId<D>, G> + Default,
    {
        let mut store = C::GradientStore::default();
        store.insert(gid, gradient);

        let mut heap = BinaryHeap::with_capacity(self.nodes.len());
        heap.push(gid.inner);
        let mut guard = gid.inner.next_id();

        while let Some(id) = heap.pop() {
            if id < guard {
                guard = id;
                let node = self
                    .nodes
                    .get(id)
                    .ok_or_else(|| Error::missing_node(func_name!()))?;
                if let Some(update_func) = &node.update_func {
                    update_func(graph, &mut store, id)?;
                }
                for input in &node.inputs {
                    if let Some(id) = input {
                        heap.push(*id);
                    }
                }
            }
        }
        Ok(store)
    }

    #[inline]
    fn do_compute_gradients_once<D, G>(
        mut self,
        graph: &mut C::GradientAlgebra,
        gid: GradientId<D>,
        gradient: G,
    ) -> Result<C::GradientStore>
    where
        C::GradientAlgebra: CoreAlgebra<D, Value = G>,
        C::GradientStore: GradientStore<GradientId<D>, G> + Default,
    {
        let mut store = C::GradientStore::default();
        store.insert(gid, gradient);

        let mut heap = BinaryHeap::with_capacity(self.nodes.len());
        heap.push(gid.inner);
        let mut guard = gid.inner.next_id();

        while let Some(id) = heap.pop() {
            if id < guard {
                guard = id;
                let node = self
                    .nodes
                    .get_mut(id)
                    .ok_or_else(|| Error::missing_node(func_name!()))?;
                if let Some(update_func) = &node.update_func {
                    update_func(graph, &mut store, id)?;
                }
                for input in &node.inputs {
                    if let Some(id) = input {
                        heap.push(*id);
                    }
                }
                node.clear();
            }
        }
        Ok(store)
    }
}

/// First order only (this is the most common case)
impl<E: Default + Clone> Graph<Config1<E>> {
    /// Propagate gradients backward, starting with the node `id`.
    /// * Allow the graph to be re-used.
    /// * Gradients are stored as pure data.
    pub fn evaluate_gradients<T>(
        &self,
        id: GradientId<T>,
        gradient: T,
    ) -> Result<GenericGradientMap1>
    where
        E: CoreAlgebra<T, Value = T>,
        T: 'static,
    {
        let mut eval = self.eval.clone();
        self.do_compute_gradients(&mut eval, id, gradient)
    }

    /// Propagate gradients backward, starting with the node `id`.
    /// * Clean up memory when possible and consume the graph.
    /// * Gradients are stored as pure data.
    pub fn evaluate_gradients_once<T>(
        self,
        id: GradientId<T>,
        gradient: T,
    ) -> Result<GenericGradientMap1>
    where
        E: CoreAlgebra<T, Value = T>,
        T: 'static,
    {
        let mut eval = self.eval.clone();
        self.do_compute_gradients_once(&mut eval, id, gradient)
    }
}

/// Higher order differentials.
impl<E: Default + Clone> Graph<ConfigN<E>> {
    /// Propagate gradients backward, starting with the node `id`.
    /// * Gradients are computed as graph values that can be differentiated later.
    /// * The graph is augmented with the nodes corresponding to gradient computations.
    pub fn compute_gradients<D>(
        &mut self,
        id: GradientId<D>,
        gradient: Value<D>,
    ) -> Result<GenericGradientMapN>
    where
        Self: CoreAlgebra<D, Value = Value<D>>,
        D: 'static,
    {
        let current = self.clone();
        current.do_compute_gradients_once(self, id, gradient)
    }
}

impl<C: Config> Clone for Graph<C> {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            eval: self.eval.clone(),
        }
    }
}

impl<C: Config> std::fmt::Debug for Graph<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        for (id, node) in self.nodes.iter() {
            write!(f, "{:?} <- {:?}; ", id, node.inputs)?;
        }
        Ok(())
    }
}
