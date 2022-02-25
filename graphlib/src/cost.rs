use std::{
    fmt,
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign, Div, Mul, Sub},
};

use rand::{
    distributions::uniform::{SampleBorrow, SampleUniform, UniformInt, UniformSampler},
    Rng,
};

use serde::{Deserialize, Serialize};

use crate::ExpRoundable;

/// The cost of an edge in a graph.
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct Cost(usize);

impl Cost {
    pub fn new(cost: usize) -> Self {
        Cost(cost)
    }

    pub fn max() -> Self {
        Cost(usize::MAX)
    }

    pub fn as_float(&self) -> f64 {
        self.0 as f64
    }

    pub fn get_usize(&self) -> usize {
        self.0
    }
}

impl ExpRoundable for Cost {
    fn to_exp_rounded(&self) -> Self {
        let c = self.0;
        if c == 0 || c == usize::MAX {
            return Cost::new(c);
        }
        let next_power = c.next_power_of_two();
        if c == next_power {
            return Cost::new(c);
        }
        let exp = (next_power as f64).log2() as u32;
        let prev_power = 2usize.pow(exp - 1);
        if next_power - c >= c - prev_power {
            Cost::new(prev_power)
        } else {
            Cost::new(next_power)
        }
    }
}

impl Add for Cost {
    type Output = Self;
    fn add(self, rhs: Cost) -> Self::Output {
        Cost(self.0 + rhs.0)
    }
}

impl Sum<Cost> for Cost {
    fn sum<I: Iterator<Item = Cost>>(iter: I) -> Self {
        iter.fold(Cost::new(0), |a, b| Cost::new(a.0 + b.0))
    }
}

impl<'a> Sum<&'a Cost> for Cost {
    fn sum<I: Iterator<Item = &'a Cost>>(iter: I) -> Self {
        iter.fold(Cost::new(0), |a, b| Cost::new(a.0 + b.0))
    }
}

impl Sub for Cost {
    type Output = Self;
    fn sub(self, rhs: Cost) -> Self::Output {
        Cost(self.0 - rhs.0)
    }
}

impl Mul<usize> for Cost {
    type Output = Self;
    fn mul(self, rhs: usize) -> Self::Output {
        Cost(self.0 * rhs)
    }
}

impl Mul<f32> for Cost {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Cost((self.0 as f32 * rhs).floor() as usize)
    }
}

impl Mul<f64> for Cost {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        Cost((self.0 as f64 * rhs).floor() as usize)
    }
}

impl Div<f32> for Cost {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        Cost((self.0 as f32 / rhs).floor() as usize)
    }
}

impl Div<f64> for Cost {
    type Output = Self;
    fn div(self, rhs: f64) -> Self::Output {
        Cost((self.0 as f64 / rhs).floor() as usize)
    }
}

impl Div<usize> for Cost {
    type Output = Self;
    fn div(self, rhs: usize) -> Self::Output {
        Cost(self.0 / rhs)
    }
}

impl AddAssign for Cost {
    fn add_assign(&mut self, rhs: Cost) {
        *self = Cost(self.0 + rhs.0)
    }
}

impl Display for Cost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for Cost {
    fn from(cost: usize) -> Self {
        Cost::new(cost)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UniformCost(UniformInt<usize>);

impl UniformSampler for UniformCost {
    type X = Cost;
    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        UniformCost(UniformInt::<usize>::new(low.borrow().0, high.borrow().0))
    }
    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        UniformSampler::new(low, high)
    }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        Cost::new(self.0.sample(rng))
    }
}

impl SampleUniform for Cost {
    type Sampler = UniformCost;
}

#[cfg(test)]
mod test_cost {
    use super::*;

    #[test]
    fn test_cost_rounding() {
        assert_eq!(Cost::new(0).to_exp_rounded(), 0.into());
        assert_eq!(Cost::new(1).to_exp_rounded(), 1.into());
        assert_eq!(Cost::new(2).to_exp_rounded(), 2.into());
        assert_eq!(Cost::new(3).to_exp_rounded(), 2.into());
        assert_eq!(Cost::new(6).to_exp_rounded(), 4.into());
        assert_eq!(Cost::new(7).to_exp_rounded(), 8.into());
        assert_eq!(Cost::max().to_exp_rounded(), Cost::max());
    }
}
