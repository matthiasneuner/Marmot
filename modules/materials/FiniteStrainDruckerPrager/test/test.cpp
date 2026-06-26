/* ---------------------------------------------------------------------
 *                                       _
 *  _ __ ___   __ _ _ __ _ __ ___   ___ | |_
 * | '_ ` _ \ / _` | '__| '_ ` _ \ / _ \| __|
 * | | | | | | (_| | |  | | | | | | (_) | |_
 * |_| |_| |_|\__,_|_|  |_| |_| |_|\___/ \__|
 *
 * Unit of Strength of Materials and Structural Analysis
 * University of Innsbruck
 * 2026 - today
 *
 * festigkeitslehre@uibk.ac.at
 *
 * This file is part of the MAteRialMOdellingToolbox (marmot).
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of marmot.
 * ---------------------------------------------------------------------
 */

#include "Marmot/FiniteStrainDruckerPrager.h"
#include "Marmot/MarmotFastorTensorBasics.h"
#include "Marmot/MarmotMaterialFiniteStrain.h"
#include "Marmot/MarmotMaterialPointSolverFiniteStrain.h"
#include "Marmot/MarmotTesting.h"
#include <string>

using namespace Marmot::Testing;
using namespace Marmot::Materials;
using namespace Marmot::FastorStandardTensors;
using namespace Marmot::FastorIndices;

void testSetup( const std::string& testName,
                const Tensor33d&   inputF,
                bool               ObjectivityCheck = false,
                bool               IsotropyCheck    = false )
{
  // Properties: K, G, cohesion, phi, psi, A, hExpDelta, hExp, H, density
  std::array< double, 10 > materialProperties_ = { 175000, 80800, 200, 30, 10, 1.0, 300, 15, 50, 0.0 };
  const double             nMaterialProperties = 10;
  const int                elLabel             = 1;

  FiniteStrainDruckerPrager mat = FiniteStrainDruckerPrager( &materialProperties_[0], nMaterialProperties, elLabel );

  double                   alphaP = 0.0;
  Tensor33d                Fp     = Marmot::FastorStandardTensors::Spatial3D::I;
  std::array< double, 10 > stateVars_;
  std::memcpy( stateVars_.data(), Fp.data(), 9 * sizeof( double ) );
  stateVars_[9] = alphaP;

  std::array< double, 10 > stateVarsInitial = stateVars_;

  mat.initializeYourself( stateVars_.data(), 10 );

  FiniteStrainDruckerPrager::Deformation< 3 > def;
  FiniteStrainDruckerPrager::TimeIncrement    timeInc = { 0, 0.1 };

  FiniteStrainDruckerPrager::ConstitutiveResponse< 3 > response( Tensor33d( 0.0 ), 0.0, 0.0, stateVars_.data() );
  FiniteStrainDruckerPrager::AlgorithmicModuli< 3 >    tangent;

  def.F = inputF;

  mat.computeStress( response, tangent, def, timeInc );

  if ( ObjectivityCheck == false && IsotropyCheck == false ) {
    for ( int i = 0; i < 3; i++ )
      for ( int j = 0; j < 3; j++ )
        throwExceptionOnFailure( checkIfEqual( response.tau( i, j ), response.tau( j, i ), 1e-10 ),
                                 testName +
                                   " - Kirchhoff stress tensor symmetry check failed "
                                   "for FiniteStrainDruckerPrager in " +
                                   std::string( __PRETTY_FUNCTION__ ) );
  }

  if ( ObjectivityCheck ) {
    Tensor33d stressUnrotated = response.tau;
    Tensor33d F_unrotated     = def.F;

    for ( int phi_deg = 0; phi_deg <= 180; phi_deg += 30 ) {
      stateVars_ = stateVarsInitial;
      mat.initializeYourself( stateVars_.data(), 10 );

      double phi = Marmot::Math::degToRad( phi_deg );

      Tensor33d Q( 0.0 );
      Q( 0, 0 ) = cos( phi );
      Q( 0, 1 ) = -sin( phi );
      Q( 1, 0 ) = sin( phi );
      Q( 1, 1 ) = cos( phi );
      Q( 2, 2 ) = 1;

      Tensor33d F_rotated = einsum< ik, kj, to_ij >( Q, F_unrotated );
      def.F               = F_rotated;

      mat.computeStress( response, tangent, def, timeInc );

      Tensor33d stressNew     = response.tau;
      Tensor33d stressRotated = einsum< iI, IJ, jJ, to_ij >( Q, stressUnrotated, Q );

      throwExceptionOnFailure( checkIfEqual( stressNew, stressRotated, 1e-10 ),
                               testName +
                                 " - Objectivity test failed "
                                 "(phi_deg=" +
                                 std::to_string( phi_deg ) + ") for FiniteStrainDruckerPrager in " +
                                 std::string( __PRETTY_FUNCTION__ ) );
    }
  }

  if ( IsotropyCheck ) {
    Tensor33d stressUnrotated = response.tau;
    Tensor33d F_unrotated     = def.F;
    Tensor33d Fp_unrotated( 0.0 );
    std::memcpy( Fp_unrotated.data(), stateVars_.data(), 9 * sizeof( double ) );
    double alphaP_unrotated = stateVars_[9];

    for ( int phi_deg = 0; phi_deg <= 180; phi_deg += 30 ) {
      double phi = Marmot::Math::degToRad( phi_deg );

      Tensor33d Q( 0.0 );
      Q( 0, 0 ) = cos( phi );
      Q( 0, 1 ) = -sin( phi );
      Q( 1, 0 ) = sin( phi );
      Q( 1, 1 ) = cos( phi );
      Q( 2, 2 ) = 1;

      Tensor33d F_rotated  = einsum< ik, kj, to_ij >( F_unrotated, Q );
      Tensor33d Fp_rotated = einsum< ik, kj, to_ij >( Fp_unrotated, Q );

      def.F = F_rotated;
      std::memcpy( stateVars_.data(), Fp_rotated.data(), 9 * sizeof( double ) );
      stateVars_[9] = alphaP_unrotated;

      mat.computeStress( response, tangent, def, timeInc );

      Tensor33d stressNew = response.tau;

      throwExceptionOnFailure( checkIfEqual( stressNew, stressUnrotated, 1e-10 ),
                               testName +
                                 " - Isotropy test failed "
                                 "(phi_deg=" +
                                 std::to_string( phi_deg ) + ") for FiniteStrainDruckerPrager in " +
                                 std::string( __PRETTY_FUNCTION__ ) );
    }
  }
}

void testUndeformedResponse()
{
  Tensor33d inputF = Marmot::FastorStandardTensors::Spatial3D::I;
  testSetup( "Undeformed configuration", inputF );
}

void testDeformationResponse()
{
  Tensor33d inputF = Marmot::FastorStandardTensors::Spatial3D::I;
  inputF( 1, 0 ) += 0.02;
  testSetup( "Simple shear deformation", inputF );
}

void testObjectivityAndIsotropy()
{
  Tensor33d inputF = Marmot::FastorStandardTensors::Spatial3D::I;
  inputF( 0, 0 )   = 1.01;
  inputF( 0, 1 )   = 0.06;
  inputF( 0, 2 )   = -0.03;
  inputF( 1, 0 )   = 0.06;
  inputF( 1, 1 )   = 1.02;
  inputF( 1, 2 )   = 0.04;
  inputF( 2, 0 )   = -0.03;
  inputF( 2, 1 )   = 0.04;
  inputF( 2, 2 )   = 0.95;

  testSetup( "Objectivity check", inputF, true, false );
  testSetup( "Isotropy check", inputF, false, true );
}

void testWithMPSolver()
{
  using namespace Marmot::Solvers;
  auto        materialProperties = std::vector< double >{ 175000, 80800, 200, 30, 10, 1.0, 300, 15, 50, 0.0 };
  auto        solveropts         = MarmotMaterialPointSolverFiniteStrain::SolverOptions();
  std::string matName            = "FINITESTRAINDRUCKERPRAGER";
  auto        solver             = MarmotMaterialPointSolverFiniteStrain( matName,
                                                       materialProperties.data(),
                                                       materialProperties.size(),
                                                       solveropts );

  MarmotMaterialPointSolverFiniteStrain::Step step;
  step.gradUIncrementTarget         = Tensor33d( 0.0 );
  step.gradUIncrementTarget( 0, 1 ) = 0.01;
  step.stressIncrementTarget        = Tensor33d( 0.0 );

  step.isGradUComponentControlled         = Tensor33t< bool >( false );
  step.isGradUComponentControlled( 0, 1 ) = true;
  step.isGradUComponentControlled( 1, 0 ) = true;

  step.isStressComponentControlled         = Tensor33t< bool >( true );
  step.isStressComponentControlled( 0, 1 ) = false;
  step.isStressComponentControlled( 1, 0 ) = false;
  step.timeStart                           = 0.0;
  step.timeEnd                             = 1;
  step.dTStart                             = .1;
  step.dTMax                               = 1;
  step.dTMin                               = 0.1;

  solver.addStep( step );
  solver.solve();
}

int main()
{
  auto tests = std::vector< std::function< void() > >{ testUndeformedResponse,
                                                       testDeformationResponse,
                                                       testObjectivityAndIsotropy,
                                                       testWithMPSolver };

  executeTestsAndCollectExceptions( tests );

  return 0;
}
