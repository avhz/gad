pub use crate::{
    algebras::{
        analytic::AnalyticAlgebra,
        arithmetic::ArithAlgebra,
        array::ArrayAlgebra,
        array_compare::ArrayCompareAlgebra,
        compare::CompareAlgebra,
        const_arith::ConstArithAlgebra,
        core::{CoreAlgebra, HasDims},
        linked::LinkedAlgebra,
        matrix::{MatProp, MatrixAlgebra},
    },
    check::Check,
    config::{Config, Config1, ConfigN},
    error::{check_equal_dimensions, Error, Result},
    eval::Eval,
    func_name,
    graph::Graph,
    networks::net::{
        CheckNet as _, ConstantData, EvalNet as _, HasGradientId, HasGradientReader, InputData,
        Net, WeightData, WeightOps,
    },
    networks::net_ext::{DiffNet as _, SingleOutputNet as _},
    number::Number,
    store::{GradientId, GradientReader, GradientStore},
    value::Value,
    Graph1, GraphN,
};
// pub use thiserror::Error as _;

#[cfg(feature = "arrayfire")]
pub use crate::algebras::arrayfire::{testing, AfAlgebra, Float, FullAlgebra};
