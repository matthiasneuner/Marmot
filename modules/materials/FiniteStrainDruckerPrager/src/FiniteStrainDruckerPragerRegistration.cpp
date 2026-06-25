#include "Marmot/FiniteStrainDruckerPrager.h"
#include "Marmot/MarmotMaterialFiniteStrainFactory.h"
#include "Marmot/MarmotMaterialFiniteStrainSubstepped.h"

namespace Marmot::Materials {

  namespace Registration {

    using namespace MarmotLibrary;

    const static bool FiniteStrainDruckerPragerRegistered = MarmotMaterialFiniteStrainFactory::registerMaterial<
      FiniteStrainDruckerPrager >( "FINITESTRAINDRUCKERPRAGER" );

    // Register the SUBSTEPPED Drucker-Prager model
    const static bool FiniteStrainDruckerPragerSubsteppedRegistered = MarmotMaterialFiniteStrainFactory::
      registerMaterial< MarmotMaterialFiniteStrainSubstepped< FiniteStrainDruckerPrager > >(
        "FINITESTRAINDRUCKERPRAGER_SUBSTEPPED" );

  } // namespace Registration
} // namespace Marmot::Materials
