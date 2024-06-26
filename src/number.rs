/// Supported numbers for default algebras.
pub trait Number:
    crate::reserved::private::Reserved
    + num::Num
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::fmt::Debug
    + serde::Serialize
    + serde::de::DeserializeOwned
    + 'static
    + Clone
    + Copy
    + Send
    + Sync
{
}
impl Number for i8 {}
impl Number for i16 {}
impl Number for i32 {}
impl Number for i64 {}
impl Number for f32 {}
impl Number for f64 {}
impl Number for num::complex::Complex<f32> {}
impl Number for num::complex::Complex<f64> {}
impl Number for num::Rational32 {}
impl Number for num::Rational64 {}
