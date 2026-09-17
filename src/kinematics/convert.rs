//! Conversions between `micro_sp`'s transform types and `k`'s.
//!
//! These exist because `k` 0.32 and `micro_sp` are built against **different
//! versions of nalgebra** (0.30 and 0.32), so `Isometry3<f64>` is two unrelated
//! types as far as the compiler is concerned and `micro_sp`'s own
//! `sp_transform_to_isometry` / `isometry_to_sp_transform` cannot be handed to the
//! IK solver. Going through plain `f64` components sidesteps the version split
//! entirely. `ur_redis_driver` carries the same pair for the same reason.

use k::nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};
use micro_sp::{SPRotation, SPTransform, SPTranslation};
use ordered_float::OrderedFloat;

pub fn sp_transform_to_isometry(transform: &SPTransform) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::new(
            transform.translation.x.into_inner(),
            transform.translation.y.into_inner(),
            transform.translation.z.into_inner(),
        ),
        UnitQuaternion::from_quaternion(Quaternion::new(
            transform.rotation.w.into_inner(),
            transform.rotation.x.into_inner(),
            transform.rotation.y.into_inner(),
            transform.rotation.z.into_inner(),
        )),
    )
}

pub fn isometry_to_sp_transform(isometry: Isometry3<f64>) -> SPTransform {
    let translation: &Vector3<f64> = &isometry.translation.vector;
    let rotation: &Quaternion<f64> = isometry.rotation.quaternion();

    SPTransform {
        translation: SPTranslation {
            x: OrderedFloat(translation.x),
            y: OrderedFloat(translation.y),
            z: OrderedFloat(translation.z),
        },
        rotation: SPRotation {
            w: OrderedFloat(rotation.w),
            x: OrderedFloat(rotation.i),
            y: OrderedFloat(rotation.j),
            z: OrderedFloat(rotation.k),
        },
    }
}
