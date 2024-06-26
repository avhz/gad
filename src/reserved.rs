pub(crate) mod private {
    /// "Sealed" trait used to avoid conflicting trait implementations.
    pub trait Reserved {}

    impl Reserved for i8 {}
    impl Reserved for i16 {}
    impl Reserved for i32 {}
    impl Reserved for i64 {}
    impl Reserved for f32 {}
    impl Reserved for f64 {}
    impl Reserved for num::complex::Complex<f32> {}
    impl Reserved for num::complex::Complex<f64> {}
    impl Reserved for num::Rational32 {}
    impl Reserved for num::Rational64 {}
}
