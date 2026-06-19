#include "Marmot/GeneralGradientEnhancedDisplacementFiniteElement.h"
#include "Marmot/MarmotElementProperty.h"
#include "Marmot/MarmotFiniteElement.h"
#include "Marmot/MarmotTesting.h"
#include "Marmot/NaturallyStabilizedGeneralGradientEnhancedDisplacementFiniteElement.h"

using namespace Marmot;
using namespace Marmot::Elements;
using namespace Marmot::Testing;

void testBasicPropertiesQuad4PlaneStress()
{
  // Test static structural properties of the GCPS4 element:
  // 2D, 4-node quad, 1 nonlocal variable, plane stress
  constexpr int nDim          = 2;
  constexpr int nNodes        = 4;
  constexpr int nNonlocalVars = 1;
  const int     elId          = 1;
  const auto    intType       = FiniteElement::Quadrature::IntegrationTypes::FullIntegration;
  using ElemType              = GeneralGradientEnhancedDisplacementFiniteElement< nDim, nNodes, nNonlocalVars >;

  auto element = std::make_unique< ElemType >( elId, intType, ElemType::SectionType::PlaneStress );

  throwExceptionOnFailure( element->getNNodes() == nNodes,
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect nNodes" );
  throwExceptionOnFailure( element->getNSpatialDimensions() == nDim,
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect nDim" );

  // sizeLoadVector = nNodes*nDim + nNodes*nNonlocalVars = 4*2 + 4*1 = 12
  throwExceptionOnFailure( element->getNDofPerElement() == 12,
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect nDof" );

  throwExceptionOnFailure( element->getElementShape() == "quad4",
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect element shape" );

  // Full integration of Quad4 uses 2x2 = 4 Gauss points
  throwExceptionOnFailure( element->getNumberOfQuadraturePoints() == 4,
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect number of QPs" );
}

void testBasicPropertiesQuad8PlaneStrain()
{
  // Test static structural properties of the GCPE8 element:
  // 2D, 8-node serendipity quad, 1 nonlocal variable, plane strain
  constexpr int nDim          = 2;
  constexpr int nNodes        = 8;
  constexpr int nNonlocalVars = 1;
  const int     elId          = 2;
  const auto    intType       = FiniteElement::Quadrature::IntegrationTypes::FullIntegration;
  using ElemType              = GeneralGradientEnhancedDisplacementFiniteElement< nDim, nNodes, nNonlocalVars >;

  auto element = std::make_unique< ElemType >( elId, intType, ElemType::SectionType::PlaneStrain );

  throwExceptionOnFailure( element->getNNodes() == nNodes,
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect nNodes" );
  throwExceptionOnFailure( element->getNSpatialDimensions() == nDim,
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect nDim" );

  // sizeLoadVector = 8*2 + 8*1 = 24
  throwExceptionOnFailure( element->getNDofPerElement() == 24,
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect nDof" );

  throwExceptionOnFailure( element->getElementShape() == "quad8",
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect element shape" );

  // Full integration of Quad8 uses 3x3 = 9 Gauss points
  throwExceptionOnFailure( element->getNumberOfQuadraturePoints() == 9,
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect number of QPs" );
}

void testBasicPropertiesHex8Solid()
{
  // Test static structural properties of the GC3D8 element:
  // 3D, 8-node hex, 1 nonlocal variable, solid
  constexpr int nDim          = 3;
  constexpr int nNodes        = 8;
  constexpr int nNonlocalVars = 1;
  const int     elId          = 3;
  const auto    intType       = FiniteElement::Quadrature::IntegrationTypes::FullIntegration;
  using ElemType              = GeneralGradientEnhancedDisplacementFiniteElement< nDim, nNodes, nNonlocalVars >;

  auto element = std::make_unique< ElemType >( elId, intType, ElemType::SectionType::Solid );

  throwExceptionOnFailure( element->getNNodes() == nNodes,
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect nNodes" );
  throwExceptionOnFailure( element->getNSpatialDimensions() == nDim,
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect nDim" );

  // sizeLoadVector = 8*3 + 8*1 = 32
  throwExceptionOnFailure( element->getNDofPerElement() == 32,
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect nDof" );

  throwExceptionOnFailure( element->getElementShape() == "hexa8",
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect element shape" );

  // Full integration of Hex8 uses 2x2x2 = 8 Gauss points
  throwExceptionOnFailure( element->getNumberOfQuadraturePoints() == 8,
                           MakeString() << __PRETTY_FUNCTION__ << ": incorrect number of QPs" );
}

void testDofIndicesPermutationPatternQuad4()
{
  // Permutation pattern for GCPS4: maps from split [u-block | k-block] to
  // interleaved [u1x, u1y, k1, u2x, u2y, k2, ...] DOF layout used in the solver
  // For nDim=2, nNodes=4, nNonlocalVars=1, nNonLocalNodes=4:
  // Expected: {0, 1, 3, 4, 6, 7, 9, 10, 2, 5, 8, 11}
  constexpr int nDim          = 2;
  constexpr int nNodes        = 4;
  constexpr int nNonlocalVars = 1;
  const int     elId          = 1;
  using ElemType              = GeneralGradientEnhancedDisplacementFiniteElement< nDim, nNodes, nNonlocalVars >;

  auto element = std::make_unique< ElemType >( elId,
                                               FiniteElement::Quadrature::IntegrationTypes::FullIntegration,
                                               ElemType::SectionType::PlaneStress );

  const std::vector< int > pattern  = element->getDofIndicesPermutationPattern();
  const std::vector< int > expected = { 0, 1, 3, 4, 6, 7, 9, 10, 2, 5, 8, 11 };

  throwExceptionOnFailure( pattern.size() == expected.size(),
                           MakeString() << __PRETTY_FUNCTION__ << ": pattern size mismatch" );
  for ( std::size_t i = 0; i < expected.size(); ++i ) {
    throwExceptionOnFailure( pattern[i] == expected[i],
                             MakeString() << __PRETTY_FUNCTION__ << ": pattern[" << i << "] mismatch" );
  }
}

void testDofIndicesPermutationPatternQuad4TwoNonlocalVars()
{
  // Permutation pattern for G2GCPS4: nDim=2, nNodes=4, nNonlocalVars=2
  // Expected: {0, 1, 4, 5, 8, 9, 12, 13, 2, 6, 10, 14, 3, 7, 11, 15}
  constexpr int nDim          = 2;
  constexpr int nNodes        = 4;
  constexpr int nNonlocalVars = 2;
  const int     elId          = 1;
  using ElemType              = GeneralGradientEnhancedDisplacementFiniteElement< nDim, nNodes, nNonlocalVars >;

  auto element = std::make_unique< ElemType >( elId,
                                               FiniteElement::Quadrature::IntegrationTypes::FullIntegration,
                                               ElemType::SectionType::PlaneStress );

  const std::vector< int > pattern  = element->getDofIndicesPermutationPattern();
  const std::vector< int > expected = { 0, 1, 4, 5, 8, 9, 12, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

  throwExceptionOnFailure( pattern.size() == expected.size(),
                           MakeString() << __PRETTY_FUNCTION__ << ": pattern size mismatch" );
  for ( std::size_t i = 0; i < expected.size(); ++i ) {
    throwExceptionOnFailure( pattern[i] == expected[i],
                             MakeString() << __PRETTY_FUNCTION__ << ": pattern[" << i << "] mismatch" );
  }
}

void testNodeFieldsOneNonlocalVar()
{
  // Node fields for nDim=2, nNodes=4, nNonlocalVars=1:
  // every node should have {"displacement", "nonlocal damage"}
  constexpr int nDim          = 2;
  constexpr int nNodes        = 4;
  constexpr int nNonlocalVars = 1;
  using ElemType              = GeneralGradientEnhancedDisplacementFiniteElement< nDim, nNodes, nNonlocalVars >;

  auto element = std::make_unique< ElemType >( 1,
                                               FiniteElement::Quadrature::IntegrationTypes::FullIntegration,
                                               ElemType::SectionType::PlaneStress );

  const auto nodeFields = element->getNodeFields();
  throwExceptionOnFailure( static_cast< int >( nodeFields.size() ) == nNodes,
                           MakeString() << __PRETTY_FUNCTION__ << ": nodeFields size mismatch" );
  for ( int i = 0; i < nNodes; ++i ) {
    throwExceptionOnFailure( nodeFields[i].size() == 2,
                             MakeString() << __PRETTY_FUNCTION__ << ": node " << i << " should have 2 fields" );
    throwExceptionOnFailure( nodeFields[i][0] == "displacement",
                             MakeString() << __PRETTY_FUNCTION__ << ": node " << i << " first field != displacement" );
    throwExceptionOnFailure( nodeFields[i][1] == "nonlocal damage",
                             MakeString()
                               << __PRETTY_FUNCTION__ << ": node " << i << " second field != nonlocal damage" );
  }
}

void testCoordinatesAtCenter()
{
  // For a unit square element [0,1]x[0,1], the center should be at (0.5, 0.5)
  constexpr int nDim          = 2;
  constexpr int nNodes        = 4;
  constexpr int nNonlocalVars = 1;
  using ElemType              = GeneralGradientEnhancedDisplacementFiniteElement< nDim, nNodes, nNonlocalVars >;

  auto element = std::make_unique< ElemType >( 1,
                                               FiniteElement::Quadrature::IntegrationTypes::FullIntegration,
                                               ElemType::SectionType::PlaneStress );

  const std::vector< double > nodeCoords = { 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 };
  element->assignNodeCoordinates( nodeCoords.data() );

  const auto centerCoords = element->getCoordinatesAtCenter();
  throwExceptionOnFailure( static_cast< int >( centerCoords.size() ) == nDim,
                           MakeString() << __PRETTY_FUNCTION__ << ": wrong dimension of center coordinates" );
  throwExceptionOnFailure( checkIfEqual( centerCoords[0], 0.5 ),
                           MakeString() << __PRETTY_FUNCTION__ << ": wrong x-coordinate at center" );
  throwExceptionOnFailure( checkIfEqual( centerCoords[1], 0.5 ),
                           MakeString() << __PRETTY_FUNCTION__ << ": wrong y-coordinate at center" );
}

void testCoordinatesAtQuadraturePoints()
{
  // For a unit square, all 4 QPs must lie strictly inside [0,1]x[0,1]
  constexpr int nDim          = 2;
  constexpr int nNodes        = 4;
  constexpr int nNonlocalVars = 1;
  using ElemType              = GeneralGradientEnhancedDisplacementFiniteElement< nDim, nNodes, nNonlocalVars >;

  auto element = std::make_unique< ElemType >( 1,
                                               FiniteElement::Quadrature::IntegrationTypes::FullIntegration,
                                               ElemType::SectionType::PlaneStress );

  const std::vector< double > nodeCoords = { 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0 };
  element->assignNodeCoordinates( nodeCoords.data() );

  const auto qpCoords = element->getCoordinatesAtQuadraturePoints();
  throwExceptionOnFailure( static_cast< int >( qpCoords.size() ) == element->getNumberOfQuadraturePoints(),
                           MakeString() << __PRETTY_FUNCTION__ << ": wrong number of QP coordinate sets" );

  for ( const auto& coords : qpCoords ) {
    throwExceptionOnFailure( static_cast< int >( coords.size() ) == nDim,
                             MakeString() << __PRETTY_FUNCTION__ << ": wrong QP coordinate dimension" );
    throwExceptionOnFailure( coords[0] > 0.0 && coords[0] < 1.0,
                             MakeString() << __PRETTY_FUNCTION__ << ": QP x-coordinate out of element bounds" );
    throwExceptionOnFailure( coords[1] > 0.0 && coords[1] < 1.0,
                             MakeString() << __PRETTY_FUNCTION__ << ": QP y-coordinate out of element bounds" );
  }
}

void testNaturallyStabilizedGradientEnhancedC3D8R()
{
  // Element Type: NaturallyStabilizedGradientEnhancedC3D8R< 1 > (3D, 8 nodes, 1 nonlocal variable, 8 nonlocal nodes)
  constexpr int nNonlocalVars = 1;
  const int     elId          = 1;
  using ElemType              = NaturallyStabilizedGradientEnhancedC3D8R< nNonlocalVars >;

  auto element = std::make_unique< ElemType >( elId );

  // Verify getPropertyNames()
  const auto propNames = element->getPropertyNames();
  throwExceptionOnFailure( propNames.size() == 3,
                           "NaturallyStabilizedGradientEnhancedC3D8R should have 3 properties." );
  throwExceptionOnFailure( propNames[0] == "bulk viscosity", "First property name incorrect." );
  throwExceptionOnFailure( propNames[1] == "coupled mode for stabilization", "Second property name incorrect." );
  throwExceptionOnFailure( propNames[2] == "stabilize nonlocal damage field", "Third property name incorrect." );

  // Verify assignProperty()
  const double bvVal                = 0.12;
  const double coupledVal           = 0.0; // false
  const double stabilizeNonlocalVal = 0.0; // false
  element->assignProperty( "bulk viscosity", &bvVal );
  element->assignProperty( "coupled mode for stabilization", &coupledVal );
  element->assignProperty( "stabilize nonlocal damage field", &stabilizeNonlocalVal );

  // Unit cube coordinates [0,1]x[0,1]x[0,1]
  const std::vector< double > nodeCoordsVec = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                                                0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0 };
  element->assignNodeCoordinates( nodeCoordsVec.data() );

  // Material properties for AT2PhaseField: E, nu, Gc, l, density
  const static std::vector< double > matProps = { 20000.0, 0.25, 2.7, 0.01, 1.0 };
  const std::string                  matName  = "AT2PHASEFIELD";
  MarmotMaterialSection              materialSection( matName, matProps.data(), matProps.size() );

  const static std::vector< double > elPropsVec = { 1.0 };
  ElementProperties                  elProps( elPropsVec.data(), elPropsVec.size() );

  element->assignProperty( elProps );
  element->assignProperty( materialSection );

  const int nStateVarsTotal = element->getNumberOfRequiredStateVars();
  throwExceptionOnFailure( nStateVarsTotal >= 36,
                           "NaturallyStabilizedGradientEnhancedC3D8R should have at least 36 state vars." );

  std::vector< double > stateVars( nStateVarsTotal, 0.0 );
  element->assignStateVars( stateVars.data(), nStateVarsTotal );

  element->initializeYourself();

  // Basic checks
  throwExceptionOnFailure( element->getNumberOfQuadraturePoints() == 1, "GC3D8R-2S must have 1 quadrature point." );
  throwExceptionOnFailure( element->getNDofPerElement() == 32, "GC3D8R-2S must have 32 DOFs." );
  throwExceptionOnFailure( element->getElementShape() == "hexa8", "GC3D8R-2S must be hexa8 shape." );

  // Test computeYourselfExplicit with default (coupled, stabilized nonlocal damage)
  // Let's create an element with defaults (only bulk viscosity is modified)
  auto elementDefault = std::make_unique< ElemType >( elId + 10 );
  elementDefault->assignProperty( "bulk viscosity", &bvVal );
  elementDefault->assignNodeCoordinates( nodeCoordsVec.data() );
  elementDefault->assignProperty( elProps );
  elementDefault->assignProperty( materialSection );

  std::vector< double > stateVarsDefault( nStateVarsTotal, 0.0 );
  elementDefault->assignStateVars( stateVarsDefault.data(), nStateVarsTotal );
  elementDefault->initializeYourself();

  const int       nDof   = element->getNDofPerElement();
  Eigen::VectorXd QTotal = Eigen::VectorXd::Zero( nDof );
  Eigen::VectorXd dQ     = Eigen::VectorXd::Zero( nDof );
  for ( int i = 0; i < nDof; ++i ) {
    dQ( i ) = 0.005 * ( i + 1 );
  }
  Eigen::VectorXd Pe_default = Eigen::VectorXd::Zero( nDof );

  double time[2] = { 0.0, 0.0 };
  double dT      = 1.0;
  double pNewDT  = 1.0;

  elementDefault->computeYourselfExplicit( QTotal.data(), dQ.data(), Pe_default.data(), time, dT, pNewDT );

  throwExceptionOnFailure( Pe_default.norm() > 0.0, "Default residual force norm should be greater than zero." );

  // Test computeYourselfExplicit with modified settings (uncoupled, disabled nonlocal stabilization)
  Eigen::VectorXd Pe_modified = Eigen::VectorXd::Zero( nDof );
  element->computeYourselfExplicit( QTotal.data(), dQ.data(), Pe_modified.data(), time, dT, pNewDT );

  throwExceptionOnFailure( Pe_modified.norm() > 0.0, "Modified residual force norm should be greater than zero." );
  throwExceptionOnFailure( ( Pe_default - Pe_modified ).norm() > 0.0,
                           "Residual force vector should change when disabling nonlocal stabilization and coupling." );
}

int main()
{
  auto tests = std::vector< std::function< void() > >{
    testBasicPropertiesQuad4PlaneStress,
    testBasicPropertiesQuad8PlaneStrain,
    testBasicPropertiesHex8Solid,
    testDofIndicesPermutationPatternQuad4,
    testDofIndicesPermutationPatternQuad4TwoNonlocalVars,
    testNodeFieldsOneNonlocalVar,
    testCoordinatesAtCenter,
    testCoordinatesAtQuadraturePoints,
    testNaturallyStabilizedGradientEnhancedC3D8R,
  };

  executeTestsAndCollectExceptions( tests );

  return 0;
}
