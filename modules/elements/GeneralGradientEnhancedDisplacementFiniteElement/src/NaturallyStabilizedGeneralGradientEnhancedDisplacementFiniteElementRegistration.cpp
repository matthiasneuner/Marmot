#include "Marmot/MarmotElementFactory.h"
#include "Marmot/NaturallyStabilizedGeneralGradientEnhancedDisplacementFiniteElement.h"

namespace Marmot::Elements::Registration {

  using namespace MarmotLibrary;
  using namespace Marmot::FiniteElement::Quadrature;

  const static bool GC3D8R_2S_isRegistered = MarmotLibrary::MarmotElementFactory::
    registerElement( "GC3D8R-2S", []( int elementID ) -> MarmotElement* {
      return new NaturallyStabilizedGradientEnhancedC3D8R< 1 >( elementID );
    } );

} // namespace Marmot::Elements::Registration
