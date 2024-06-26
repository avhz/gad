/// The default algebra that only computes forward values.
#[derive(Clone, Default)]
pub struct Eval {
    pub(crate) check: crate::check::Check,
}

impl Eval {
    /// Access the underlying default "Check" algebra.
    #[inline]
    pub fn check(&mut self) -> &mut crate::check::Check {
        &mut self.check
    }
}
