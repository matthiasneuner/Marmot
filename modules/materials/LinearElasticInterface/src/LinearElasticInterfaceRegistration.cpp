#include "../include/Marmot/LinearElasticInterface.h"
#include "Marmot/MarmotMaterialRegistrationHelper.h"

namespace Marmot::Materials {

  namespace Registration {

    constexpr int LinearElasticInterfaceCode = 50001;

    using namespace MarmotLibrary;

    const static bool LinearElasticIsRegistered = MarmotMaterialFactory::
      registerMaterial( LinearElasticInterfaceCode,
                        "LINEARELASTICINTERFACE",
                        makeDefaultMarmotMaterialFactoryFunction< class LinearElasticInterface >() );

  } // namespace Registration
} // namespace Marmot::Materials
