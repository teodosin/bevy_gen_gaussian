// Draft file with sketches for the new refactor. Not used yet.

// The goal of this refactor is to decouple mass and form. 
// We want to be able to choose any gaussian cloud and have it
// dynamically interpolate to any other gaussian cloud. 

// Open questions:

// The entities set the relationships between the different clouds and targets.
// The compute shader does the actual interpolation and adds other effects.
// So what systems are actually required? What are the building blocks?

// Let's run through a scenario. 

use bevy::prelude::*;
use bevy_gaussian_splatting::{PlanarGaussian3dHandle};

// The two main components are Mass and Form. They can't exist on the same entity,
// so they're mutually exclusive.
#[derive(Component)]
pub struct Mass {
    pub target_form: Option<Entity>,
}

#[derive(Component)]
#[require(PlanarGaussian3dHandle)]
pub struct Form {

}

#[derive(Event)]
pub struct MassToForm {
    // Parameters for the interpolation
    pub duration: f32,
    pub ease: EaseFunction,
}

/// System to handle the interpolation from Masses to Forms.
/// Reacts to MassToForm events.
/// 
/// Will also handle one-to-many and many-to-one conversions.
/// Now that I think of, is that actually the primary purpose of this system?
/// For basic interpolation it could be enough to just extract the masses and forms
/// to the render world, so maybe what this system would do is just ensure that
/// the amounts of splats are correct and that in the event of merging or splitting
/// the correct clouds are initialised or removed.
fn mass_to_form(
    mut commands: Commands,
    mut events: EventReader<MassToForm>,
    query: Query<(Entity, &Mass)>,
) {
    
}

// Scenarios the api needs to cover for Beat Cauldron:

// One to one conversion
// Many to one conversion
// One to many conversion
// Mass whose Form is that of a rectangular fluid simulation