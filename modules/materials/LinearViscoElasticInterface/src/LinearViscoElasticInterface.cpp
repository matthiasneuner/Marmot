#include "Marmot/LinearViscoElasticInterface.h"
#include "Marmot/MarmotElasticity.h"
#include "Marmot/MarmotMaterialHypoElasticInterface.h"
#include "Marmot/MarmotJournal.h"
#include "Marmot/MarmotWiechertInterface.h"
#include "Marmot/MarmotMath.h"
#include "Marmot/MarmotTypedefs.h"
#include "Marmot/MarmotViscoelasticity.h"
#include "Marmot/MarmotUtility.h"
#include "Marmot/MarmotVoigt.h"

#include "Fastor/Fastor.h"
#include "Marmot/interface_material_helper_functions.h"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <Fastor/expressions/linalg_ops/unary_norm_op.h>
#include <Fastor/tensor/TensorMap.h>

#include "autodiff/forward/real.hpp"
#include <iostream>
#include <map>
#include <string>

using namespace Marmot;
using namespace Eigen;
using namespace Fastor;

using Tensor1D = Fastor::Tensor<double,3>;
using Tensor2D = Fastor::Tensor<double,3,3>;
using Tensor3D = Fastor::Tensor<double,3,3,3>;
using Tensor4D = Fastor::Tensor<double,3,3,3,3 >;

namespace Marmot::Materials {


  
  LinearViscoElasticInterface::LinearViscoElasticInterface( const double* materialProperties,
                                                            int           nMaterialProperties, 
                                                            int                materialNumber )
    : MarmotMaterialHypoElasticInterface( materialProperties, nMaterialProperties, materialNumber ),
    // clang-format off 
    // elasticity parameters
    E_M                     (materialProperties[0]),
    nu_M                    (materialProperties[1]),
    E_I                     (materialProperties[2]),
    nu_I                    (materialProperties[3]),
    E_0                     (materialProperties[4]),
    nu_0                    (materialProperties[5]),
    h                       (materialProperties[6]),

    // equivalent viscoelastic parameters common for Jump u
    m_Ru                      (materialProperties[7]),
    n_Ru                      (materialProperties[8]),
    nMaxwell_Ru               (static_cast<size_t>(materialProperties[9])),
    minTau_Ru                 (materialProperties[10]),
    
    // equivalent viscoelastic parameters common for surface strain
    m_Rs                      (materialProperties[11]),
    n_Rs                      (materialProperties[12]),
    nMaxwell_Rs                (static_cast<size_t>(materialProperties[13])),
    minTau_Rs                 (materialProperties[14]),
    

    timeToDays              (materialProperties[15])
  // clang-format on
  {

    //retardationTimes_Ru = Marmot::Materials::WiechertInterface::generateRetardationTimes(nMaxwell_Ru, minTau_Ru, sqrt(10));
    //retardationTimes_Rs = Marmot::Materials::WiechertInterface::generateRetardationTimes(nMaxwell_Rs, minTau_Rs, sqrt(10));

    Eigen::VectorXd retardationTimes_Ru(2);
    retardationTimes_Ru<<0.01, 0.1;
    Eigen::VectorXd retardationTimes_Rs(2);
    retardationTimes_Rs<< 0.01, 0.1; 

    using namespace Marmot::ContinuumMechanics::Viscoelasticity ;
    auto phiRu_ = [&](autodiff::Real<powerLawApproximationOrder, double> tau){
        return ComplianceFunctions::powerLaw( tau, m_Ru, n_Ru);
    };

    auto phiRs_ = [&](autodiff::Real<powerLawApproximationOrder, double> tau){
        return ComplianceFunctions::powerLaw( tau, m_Rs, n_Rs);
    };

    //elasticModuli_Ru = Marmot::Materials::WiechertInterface::computeElasticModuli_Ru<powerLawApproximationOrder>(phiRu_, retardationTimes_Ru);
    //elasticModuli_Rs = Marmot::Materials::WiechertInterface::computeElasticModuli_Rs<powerLawApproximationOrder>(phiRs_, retardationTimes_Rs);
    
    //zerothWiechertStiffness_Ru = m_Ru*(1. - n_Ru )*pow( 2., n_Ru )*pow(minTau_Ru/sqrt(10.), n_Ru);  
    //zerothWiechertStiffness_Rs = m_Rs*(1. - n_Rs )*pow( 2., n_Rs )*pow(minTau_Rs/sqrt(10.), n_Rs);
    Eigen::VectorXd elasticModuli_Ru(2);
    elasticModuli_Ru<<0.01 , 0.1; 
    
    Eigen::VectorXd elasticModuli_Rs(2);
    elasticModuli_Rs<< 0.01 , 0.1; 
    
    Eigen::VectorXd zerothWiechertStiffness_Ru(2);
    zerothWiechertStiffness_Ru<< 1. , 1.;
    Eigen::VectorXd zerothWiechertStiffness_Rs(2);
    zerothWiechertStiffness_Rs<< 1. , 1.;

  }

  void LinearViscoElasticInterface::computeStress( double*  force,
                                                   double*  surface_stress,
                                                   double* dStress_dStrain,
                                                   const double* dU,
                                                   const double* dSurface_strain,
                                                   const double* normal,
                                                   const double* timeOld,
                                                   const double  dT,
                                                   double&       pNewDT)
  {
    // map to force, surface stress, displacement, surface strain, normal and tangent stiffness 
    // use Fastor because we really need to use the einsum 
    //std::cout<<"Inside proper file\n";
     
    Fastor::Tensor<double, 3>  force_ftensor(force);
    Fastor::Tensor<double,3,3>  surface_stress_ftensor(surface_stress);
    Fastor::Tensor<double,21,21> dStress_dStrain_ftensor(dStress_dStrain);
    auto dU_ftensor_const = Fastor::TensorMap<const double,6,1>(dU);
    auto dSurface_strain_ftensor_const = Fastor::TensorMap<const double,18,1>(dSurface_strain);
    auto normal_ftensor_const = Fastor::TensorMap<const double, 3>(normal); 
    
    Fastor::Tensor<double,6,1> dU_ftensor(dU_ftensor_const.data());
    Fastor::Tensor<double,18,1> dSurface_strain_ftensor(dSurface_strain_ftensor_const.data());
    Fastor::Tensor<double,3> normal_ftensor(normal_ftensor_const.data());

    auto [Z_ijkl, H_inv_ij, H_inv_nF_ijk, Yn_H_inv_Fn_ijkl] = calculate_interface_material_parameters(normal_ftensor, E_M, nu_M, E_I, nu_I, E_0, nu_0); 
    
    //Assign the material matrices to a larger structure. (Not necessary ...)
    Eigen::Matrix<double, 21, 21> Cel = Eigen::Matrix<double, 21,21>::Zero();

    // handle zero strain increment
    if ( Fastor::norm(dU_ftensor) <1e-14 && Fastor::norm(dSurface_strain_ftensor) <1e-14 && dT==0  ) {
        Cel.block(0,0,9,9) = convert4thOrderTensorToMatrix(h/2.*Z_ijkl);
        Cel.block(12,12,9,9) = convert4thOrderTensorToMatrix(h/2.*Yn_H_inv_Fn_ijkl);
        Cel.block(0,9,9,3) = convert3rdOrderTensorToMatrix(H_inv_nF_ijk);
        Cel.block(9,9,3,3) = convert2ndOrderTensorToMatrix(2./h*H_inv_ij);
    
        Fastor::Tensor<double,21,21> Cel_fastor(Cel.data()); 
        dStress_dStrain_ftensor = Cel_fastor;
        std::copy(dStress_dStrain_ftensor.data(), dStress_dStrain_ftensor.data() + 21*21, dStress_dStrain);
      return;
    }
    //visco elastic step
  
    Eigen::Ref< WiechertInterface::mapStateVarMatrix_Ru> creepStateVars_Ru( stateVarManager->MaxwellStateVars_Ru );
    Eigen::Ref< WiechertInterface::mapStateVarMatrix_Rs> creepStateVars_Rs( stateVarManager->MaxwellStateVars_Rs );
     
    const double dTimeDays = dT * timeToDays;
    
    Vector3d creep_Ru_Increment = Vector3d::Zero();
    double creep_Ru_stiffness = 0;

    WiechertInterface::evaluateWiechert_Ru(dTimeDays,
                                         elasticModuli_Ru,
                                         retardationTimes_Ru,
                                         creepStateVars_Ru,
                                         creep_Ru_stiffness,
                                         creep_Ru_Increment,
                                          1.0);
    
    Vector9d creep_Rs_Increment = Vector9d::Zero();
    double creep_Rs_stiffness = 0;
 
    WiechertInterface::evaluateWiechert_Rs(dTimeDays,
                                         elasticModuli_Rs,
                                         retardationTimes_Rs,
                                         creepStateVars_Rs,
                                         creep_Rs_stiffness,
                                         creep_Rs_Increment,
                                                 1.0);

    using namespace Marmot::ContinuumMechanics::Viscoelasticity;

    //Evaluate effective compliances due to the displacement jump and the surface stress

    auto [H_inv_ij_effective, Z_ijkl_effective, unitH_inv_voigt_full, unitZ_voigt_full] = calculate_effective_properties(zerothWiechertStiffness_Ru,
                                                                                                                        creep_Ru_stiffness,
                                                                                                                        zerothWiechertStiffness_Rs,
                                                                                                                        creep_Rs_stiffness,
                                                                                                                        normal_ftensor, 
                                                                                                                        E_M, 
                                                                                                                        nu_M,
                                                                                                                        E_I,
                                                                                                                        nu_I, 
                                                                                                                        E_0,
                                                                                                                        nu_0);
    
    enum {i,j,k,l};
    // Calculate jump increment
    Tensor1D jumpU_ftensor = dU_ftensor(Fastor::seq(0,3),0)-dU_ftensor(Fastor::seq(3,Fastor::last),0);    
    
    // Calculate average surface strain increment
    Fastor::Tensor<double, 9,1> average_dSurface_strain_ftensor = 1./2.*(dSurface_strain_ftensor(Fastor::seq(0,9),0)+dSurface_strain_ftensor(Fastor::seq(9,Fastor::last),0));
    auto average_dSurface_strain_ftensor_reshape = Fastor::reshape<3,3>(average_dSurface_strain_ftensor);

    Fastor::TensorMap<double,3> creep_Ru_increment_fastor(creep_Ru_Increment.data());

    //std::cout<<"creep_Ru_increment_fastor:\n"<<creep_Ru_increment_fastor<<'\n';
    Tensor1D  deltaforce_i = Fastor::einsum<Fastor::Index<i,j>, Fastor::Index<j>, Fastor::OIndex<i>>(H_inv_ij_effective, jumpU_ftensor)-creep_Ru_increment_fastor;

    Fastor::TensorMap<double,3,3> creep_Rs_increment_fastor(creep_Rs_Increment.data()); 

    //std::cout<<"creep_Rs_increment_fastor:\n"<<creep_Rs_increment_fastor<<'\n';
    Tensor2D deltasurface_stress_ij = Fastor::einsum<Fastor::Index<i,j,k,l>,Fastor::Index<k,l>, Fastor::OIndex<i,j>>(Z_ijkl_effective, average_dSurface_strain_ftensor_reshape)+creep_Rs_increment_fastor; 
    
    force_ftensor     += 2./h*deltaforce_i;
    surface_stress_ftensor += h/2.*deltasurface_stress_ij;
     
    Cel.block(0,0,9,9) = convert4thOrderTensorToMatrix(h/2.*Z_ijkl_effective);
    Cel.block(12,12,9,9) = convert4thOrderTensorToMatrix(h/2.*Yn_H_inv_Fn_ijkl);
    Cel.block(0,9,9,3) = convert3rdOrderTensorToMatrix(H_inv_nF_ijk);
    Cel.block(9,9,3,3) = convert2ndOrderTensorToMatrix(2./h*H_inv_ij_effective);

    Fastor::Tensor<double,21,21> Cel_fastor(Cel.data()); 
    dStress_dStrain_ftensor = Cel_fastor;
    
    std::copy(force_ftensor.data(), force_ftensor.data() + 3, force);
    std::copy(surface_stress_ftensor.data(), surface_stress_ftensor.data() + 3*3, surface_stress);
    std::copy(dStress_dStrain_ftensor.data(), dStress_dStrain_ftensor.data() + 21*21, dStress_dStrain);
    
    // Use already available functionality convert Fastor tensors to Eigen matricfes/vectors
    // Tranform to Eigen matrices to work with the internal machinery of KelvinChainInterface ...
    Eigen::Map<Eigen::Matrix<double,3, Eigen::RowMajor>> jumpU_voigt_full(jumpU_ftensor.data());
    Eigen::Map<Eigen::Matrix<double,9, Eigen::RowMajor>> average_dSurface_strain_ftensor_reshape_voigt_full(average_dSurface_strain_ftensor_reshape.data());
    
  
    WiechertInterface::updateStateVarMatrix_Ru( dTimeDays,
                                               elasticModuli_Ru,
                                               retardationTimes_Ru,
                                               creepStateVars_Ru,
                                               jumpU_voigt_full,
                                               unitH_inv_voigt_full);
    
    WiechertInterface::updateStateVarMatrix_Rs( dTimeDays,
                                               elasticModuli_Rs,
                                               retardationTimes_Rs,
                                               creepStateVars_Rs,
                                               average_dSurface_strain_ftensor_reshape_voigt_full,
                                               unitZ_voigt_full);  
    
    return;
      };
  
  void LinearViscoElasticInterface::assignStateVars( double* stateVars_, int nStateVars )
  {
    if ( nStateVars < getNumberOfRequiredStateVars() )
      throw std::invalid_argument( MakeString() << __PRETTY_FUNCTION__ << ": Not sufficient stateVars!" );

    this->stateVarManager = std::make_unique< LinearViscoElasticInterfaceStateVarManager >( stateVars_, nMaxwell_Ru, nMaxwell_Rs );

    MarmotMaterial::assignStateVars( stateVars_, nStateVars );
  }

  StateView LinearViscoElasticInterface::getStateView( const std::string& stateName )
  {
    return stateVarManager->getStateView( stateName );
  }

  int LinearViscoElasticInterface::getNumberOfRequiredStateVars()
  {
    return LinearViscoElasticInterfaceStateVarManager::layout.nRequiredStateVars; //+ nKelvin_Ju * 3 + nKelvin_Js * 9;
  }
} // namespace Marmot::Materials
