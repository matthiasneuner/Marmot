/* ---------------------------------------------------------------------
 *                                       _
 *  _ __ ___   __ _ _ __ _ __ ___   ___ | |_
 * | '_ ` _ \ / _` | '__| '_ ` _ \ / _ \| __|
 * | | | | | | (_| | |  | | | | | | (_) | |_
 * |_| |_| |_|\__,_|_|  |_| |_| |_|\___/ \__|
 *
 * Unit of Strength of Materials and Structural Analysis
 * University of Innsbruck,
 * 2020 - today
 *
 * festigkeitslehre@uibk.ac.at
 *
 * Alexandros Stathas alexandros.stathas@boku.ac.at
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
//#`:pragma once
//#include "Marmot/MarmotTypedefs.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/src/Core/Matrix.h>
#include <Fastor/Fastor.h>
#include <Fastor/expressions/linalg_ops/unary_norm_op.h>
#include <Fastor/expressions/linalg_ops/unary_trans_op.h>
#include <Fastor/tensor/AbstractTensorFunctions.h>
#include <Fastor/tensor_algebra/einsum.h>
#include <Fastor/tensor_algebra/indicial.h>
#include <ostream>
#include <tuple>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cassert>
#include <cmath>
#include <iostream>
#include "../include/Marmot/interface_material_helper_functions.h"
#include "../include/Marmot/MarmotElasticity.h"
#include "Marmot/MarmotTensor.h"
#include "Marmot/MarmotTypedefs.h"

using namespace Eigen;
using namespace Fastor;

using Tensor1D = Fastor::Tensor<double,3>;
using Tensor2D = Fastor::Tensor<double,3,3>;
using Tensor3D = Fastor::Tensor<double,3,3,3>;
using Tensor4D = Fastor::Tensor<double,3,3,3,3 >;

void assert_equivalent_F_Falt_Y(
    Tensor4D F ,
    Tensor4D Y ,
    Tensor4D A_0 ,
    Tensor4D L_0 ,
    Tensor4D A_M ,
    Tensor4D L_M ,
    Tensor4D A_I ,
    Tensor4D L_I 
    );

enum {a,i,b,j,k,l,m,n};

Tensor2D compute_inv( const Tensor2D& I, Tensor2D& Q )
{
  // Solve G in batch mode using LU decomposition
  Eigen::Matrix3d Im (I.data());
  Eigen::Matrix3d Qm (Q.data());
  Eigen::Matrix3d G_mat = Qm.fullPivLu().solve( Im );

  // Store result back into G (reinterpret as a 3D Fastor tensor)
  Tensor2D G(0);
  Eigen::Map< Eigen::Matrix< double, 3, 3 , Eigen::RowMajor > >( G.data() ) = G_mat;

  return G;
}
std::tuple< Tensor4D, const Tensor4D, Tensor4D, Tensor2D>interface_geometry_system_couplings( 
                                          const Tensor2D& I,
                                          const Tensor2D& N,
                                          const Tensor2D& T,
                                          const Tensor4D& L
                                         )
{

  // **Compute Q = einsum("aibjq,ijq->abq", L_expanded, N)**
  Tensor2D Q = Fastor::einsum<Fastor::Index<a,i,b,j>,Fastor::Index<i,j>,Fastor::OIndex<a,b>>(L, N);
  Tensor2D G = compute_inv(I, Q );

  Tensor4D A = Fastor::einsum<Fastor::Index<a,b>,Fastor::Index<i,j>,Fastor::OIndex<a,i,b,j>>(G, N );
  Tensor4D LA = Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(L,A);
  Tensor4D LAL = Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(LA, L );
  Tensor4D B = L - LAL;
  // check that the expressions are consistent
  //assert_consistent_arrays_indices(L, A, B, M, I);
  
  return std::make_tuple(B, L, A, G);
}

std::tuple<Tensor4D, Tensor2D, Tensor3D, Tensor4D> calculate_material_matrices(
                                 const Tensor1D& normal,
                                 const Tensor2D& I,  
                                 const Tensor2D& N, 
                                 const Tensor2D& T, 
                                 const Tensor4D& C_0_aibj,
                                 const Tensor4D& C_M_aibj,
                                 const Tensor4D& C_I_aibj
                                )
  {
    Tensor2D G_0 ;
    Tensor4D A_0 ;
    Tensor4D B_0 ;
    Tensor4D L_0 ;

    Tensor2D G_M ;
    Tensor4D A_M ;
    Tensor4D B_M ;
    Tensor4D L_M ;

    Tensor2D G_I ;
    Tensor4D A_I ;
    Tensor4D B_I ;
    Tensor4D L_I ;
    
    std::tie(B_0, L_0, A_0, G_0) = interface_geometry_system_couplings(
        I, N, T, C_0_aibj
    );

    std::tie(B_M, L_M, A_M, G_M) = interface_geometry_system_couplings(
        I, N, T, C_M_aibj
    );

    std::tie(B_I, L_I, A_I, G_I) = interface_geometry_system_couplings(
        I, N, T, C_I_aibj
    );
    
    Tensor4D F = -2.0 * Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(A_0, L_0);
    F+= Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(A_M, L_M);
    F+= Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(A_I, L_I);
    
    Tensor4D Y = Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(L_M, A_M);
    Y+= Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(L_I, A_I);
    Y-= 2.0 * Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(L_0, A_0);  

    assert_equivalent_F_Falt_Y(F , Y , A_0 , L_0 , A_M , L_M , A_I , L_I );

    Tensor2D H = 2.0 * G_0 - G_M - G_I;
    
    Tensor4D Z = B_M + B_I - 2.0 * B_0;
    Tensor2D H_inv = compute_inv(I, H );
    
    Tensor3D nF = Fastor::einsum<Fastor::Index<a>,Fastor::Index<a,i,b,j>,Fastor::OIndex<i,b,j>>(normal, F);
    Tensor3D Fn = Fastor::einsum<Fastor::Index<a,i,b,j>,Fastor::Index<j>,Fastor::OIndex<a,i,b>>(F, normal);
    Tensor3D Yn = Fastor::einsum<Fastor::Index<a,i,b,j>,Fastor::Index<j>,Fastor::OIndex<a,i,b>>(Y, normal);
    Tensor3D H_inv_nF = Fastor::einsum<Fastor::Index<a,b>,Fastor::Index<b,i,j>,Fastor::OIndex<a,i,j>>(H_inv, nF);
    Tensor4D Yn_H_inv_Fn = Fastor::einsum<Fastor::Index<a,i,m>,Fastor::Index<m,n>,Fastor::Index<n,b,j>,Fastor::OIndex<a,i,b,j>>(Yn, H_inv, Fn);
    return std::make_tuple(Z, H_inv, H_inv_nF, Yn_H_inv_Fn);
  }

void assert_equivalent_F_Falt_Y(
    Tensor4D F ,
    Tensor4D Y ,
    Tensor4D A_0 ,
    Tensor4D L_0 ,
    Tensor4D A_M ,
    Tensor4D L_M ,
    Tensor4D A_I ,
    Tensor4D L_I 
    )
{
  Tensor4D F_alt = -2.0 * Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(A_0, L_0);
    F_alt+= Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(A_M, L_M);
    F_alt+= Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(A_I, L_I);
    
    double atol = 1e-8;
    Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,1>> F_alt_flat(F_alt.data(), F_alt.size());
    Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,1>> F_flat(F.data(), F.size());
   
    double max_diff = (F_alt_flat-F_flat).cwiseAbs().maxCoeff();
    if (max_diff > atol){
      std::cerr << "Arrays are not equal within the tolerance."<< std::endl;
      std::abort();
    }
}

void assert_Q_ij_G_ij(Tensor2D Q_ij, Tensor1D normal, double E)
{
  double nu = 0.0;

  double mu = E/(2.*(1.+nu));

  Tensor2D I = {{1.0,0.0,0.0},
                  {0.0,1.0,0.0},
                  {0.0,0.0,1.0}
                 };

  Tensor4D I1_ijkl = Fastor::einsum<Fastor::Index<i,k>,Fastor::Index<j,l>,Fastor::OIndex<i,j,k,l>>(I, I);
  Tensor4D I2_ijkl = Fastor::einsum<Fastor::Index<i,l>,Fastor::Index<j,k>,Fastor::OIndex<i,j,k,l>>(I, I);
  Tensor4D IB_ijkl =I1_ijkl + I2_ijkl;
  Tensor2D N =Fastor::einsum<Fastor::Index<j>,Fastor::Index<l>,Fastor::OIndex<j,l>>(normal,normal);  
  Tensor2D Q_calc_ik = mu*Fastor::einsum<Fastor::Index<i,j,k,l>,Fastor::Index<j,l>,Fastor::OIndex<i,k>>(IB_ijkl,N);  
  
  double diff_Q_Q_calc = Fastor::norm((Q_ij-Q_calc_ik));
  std::cout<<"diff_Q_Q_calc: "<<diff_Q_Q_calc<<"\n";
  if (diff_Q_Q_calc>1e-8){
    std::cerr<<"Q, Q_calc not the same"<<std::endl;
    std::abort();  
  }
  //Tensor2D G_calc_ik = compute_inv(I, Q_calc_ik);
  Tensor2D G_calc_an_ik = 1/mu*I-1/(2*mu)*N;
  Tensor2D G_ik = compute_inv(I, Q_ij);
  double diff_G_G_calc = Fastor::norm((G_ik-G_calc_an_ik));
  std::cout<<"diff_G_G_calc: "<<diff_G_G_calc<<"\n";
  if (diff_G_G_calc>1e-8){
    std::cerr<<"G, G_calc not the same"<<std::endl;
    std::abort();  
  }
  Tensor4D A_ijkl = Fastor::einsum<Fastor::Index<i,k>, Fastor::Index<j,l>, Fastor::OIndex<i,j,k,l>>(G_ik,N); 
  Tensor4D A_calc_ijkl = (1/mu)*Fastor::einsum<Fastor::Index<i,k>,Fastor::Index<j,l>,Fastor::OIndex<i,j,k,l>>(I,N);
  A_calc_ijkl -= (1/(2*mu))*Fastor::einsum<Fastor::Index<i,k>,Fastor::Index<j,l>,Fastor::OIndex<i,j,k,l>>(N,N);
  double diff_A_A_calc = Fastor::norm((A_ijkl-A_calc_ijkl));


  std::cout<<"diff_A_A_calc: "<<diff_A_A_calc<<"\n";
  if (diff_A_A_calc>1e-8){
    std::cerr<<"A, A_calc not the same"<<std::endl;
    std::abort();  
  }
  
}

void assert_Z_ijkl(Tensor4D Z_ijkl, Tensor1D normal, double E)
  {
  //Analytical Z_tensor
  //
  double nu = 0.0;

  double mu = E/(2.*(1.+nu));

  Tensor2D I = {{1.0,0.0,0.0},
                  {0.0,1.0,0.0},
                  {0.0,0.0,1.0}
                 };
  Tensor2D N =Fastor::einsum<Fastor::Index<j>,Fastor::Index<l>,Fastor::OIndex<j,l>>(normal,normal);  

  Tensor4D Ia_ijkl = Fastor::einsum<Fastor::Index<i,k>,Fastor::Index<j,l>,Fastor::OIndex<i,j,k,l>>(I, N);
  Tensor4D Ib_ijkl = -(1./2.)*Fastor::einsum<Fastor::Index<i,k>,Fastor::Index<j, l>,Fastor::OIndex<i,j,k,l>>(N, N);
  Tensor4D Ic_ijkl = Fastor::einsum<Fastor::Index<j,k>,Fastor::Index<i,l>,Fastor::OIndex<i,j,k,l>>(I, N);
  Tensor4D Id_ijkl = -(1./2.)*Fastor::einsum<Fastor::Index<j,k>,Fastor::Index<i,l>,Fastor::OIndex<i,j,k,l>>(N, N);
  Tensor4D Ie_ijkl = Fastor::einsum<Fastor::Index<i,l>,Fastor::Index<j,k>,Fastor::OIndex<i,j,k,l>>(I, N);
  Tensor4D If_ijkl = -(1./2.)*Fastor::einsum<Fastor::Index<i,l>,Fastor::Index<j,k>,Fastor::OIndex<i,j,k,l>>(N, N);
  Tensor4D Ig_ijkl = Fastor::einsum<Fastor::Index<j,l>,Fastor::Index<i,k>,Fastor::OIndex<i,j,k,l>>(I, N);
  Tensor4D Ih_ijkl = -(1./2.)*Fastor::einsum<Fastor::Index<j,l>,Fastor::Index<i,k>,Fastor::OIndex<i,j,k,l>>(N, N);

  Tensor4D IA_ijkl =Ia_ijkl + Ib_ijkl + Ic_ijkl + Id_ijkl + Ie_ijkl + If_ijkl + Ig_ijkl + Ih_ijkl ;

  Tensor4D I1_ijkl = Fastor::einsum<Fastor::Index<i,k>,Fastor::Index<j,l>,Fastor::OIndex<i,j,k,l>>(I, I);
  Tensor4D I2_ijkl = Fastor::einsum<Fastor::Index<i,l>,Fastor::Index<j,k>,Fastor::OIndex<i,j,k,l>>(I, I);
  Tensor4D IB_ijkl =I1_ijkl + I2_ijkl;

  Tensor4D B_ijkl = mu*IB_ijkl - mu*IA_ijkl;  
  std::cout<<"B_ijkl: \n"<<B_ijkl<<"\n";
  std::cout<<"Z_ijkl: \n"<<Z_ijkl<<"\n";

  double diff_Z_B = Fastor::norm((Z_ijkl-B_ijkl));
  std::cout<<"diff_Z_B: "<<diff_Z_B<<"\n";
  if (diff_Z_B>1e-8){
    std::cerr<<"Z, B not the same"<<std::endl;
    std::abort();
  }
}

// Convert 4th-order Fastor tensor (3x3x3x3) to Eigen 9x9 matrix
Eigen::Matrix<double,9,9> convert4thOrderTensorToMatrix(const Tensor4D& tensor) {
    Eigen::Matrix<double,9,9> matrix(9, 9);
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    int row = 3 * i + j;  // Convert (i, j) to single index
                    int col = 3 * k + l;  // Convert (k, l) to single index
                    matrix(row, col) = tensor(i, j, k, l);
                }
            }
        }
    }
    
    return matrix;
}

// Convert 3rd-order Fastor tensor (3x3x3) to Eigen 9x3 matrix
Eigen::Matrix<double,9,3> convert3rdOrderTensorToMatrix(const Tensor3D& tensor) {
    Eigen::Matrix<double,9,3> matrix(9, 3);
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                int row = 3 * i + j;  // Convert (i, j) to single index
                matrix(row, k) = tensor(i, j, k);
            }
        }
    }
    
    return matrix;
}

// Convert 2nd-order Fastor tensor (3x3) to Eigen 3x3 matrix
Eigen::Matrix<double,3,3>convert2ndOrderTensorToMatrix(const Tensor2D& tensor){
    Eigen::Matrix<double,3,3> matrix(3, 3);
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
                matrix(i, j) = tensor(i, j);
            }
        }   
    return matrix;
}

Tensor4D voigtToStiffness(const Eigen::Matrix<double,6,6> &voigtStiffness)
{
  using namespace Marmot::ContinuumMechanics::TensorUtility::IndexNotation;
    
  Fastor::Tensor<double,3,3,3,3> stiffness;
  stiffness.zeros(); // Set to zero

  int row, col;
  for (int i=0; i<3; ++i)
  for (int j=0; j<3; ++j) {
    row = toVoigt<3>(i,j);
    for (int k=0; k<3; ++k)
    for (int l=0; l<3; ++l) {
        col = toVoigt<3>(k,l);
        stiffness(i,j,k,l) += voigtStiffness(row, col);
    }
  }

return stiffness;
}

// Scale factor: 1 for diagonal terms, sqrt(2) for shear
inline double voigtScaling(int i, int j) {
    return (i == j) ? 1.0 : std::sqrt(2.0);
}

// Maps compliance tensor S_{IJ} in Voigt to S_{ijkl} in full tensor form
Tensor4D voigtToCompliance(const Eigen::Matrix<double,6,6> &voigtCompliance) {
  using namespace Marmot::ContinuumMechanics::TensorUtility::IndexNotation;

  Fastor::Tensor<double,3,3,3,3> compliance;
    compliance.zeros();

    for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j) {
        int I = toVoigt<3>(i,j);
        double scale_I = voigtScaling(i,j);

        for (int k=0; k<3; ++k)
        for (int l=0; l<3; ++l) {
            int J = toVoigt<3>(k,l);
            double scale_J = voigtScaling(k,l);

            compliance(i,j,k,l) = voigtCompliance(I,J) / (scale_I * scale_J);
        }
    }

  return compliance;
}

std::tuple<Tensor4D, Tensor2D, Tensor3D, Tensor4D> calculate_interface_material_parameters(
                                                                                           const Tensor1D& normal,
                                                                                           const double& E_M,
                                                                                           const double& nu_M,
                                                                                           const double& E_I,
                                                                                           const double& nu_I,
                                                                                           const double& E_0,
                                                                                           const double& nu_0)
{   using namespace Marmot::ContinuumMechanics::Elasticity::Isotropic;
    Eigen::Matrix<double, 6, 6> C_M_voigt_full = stiffnessTensor(E_M, nu_M);
    Eigen::Matrix<double, 6, 6> C_I_voigt_full = stiffnessTensor(E_I, nu_I);
    Eigen::Matrix<double, 6, 6> C_0_voigt_full = stiffnessTensor(E_0, nu_0);

    Tensor2D I = {{1.0,0.0,0.0},
                  {0.0,1.0,0.0},
                  {0.0,0.0,1.0}
                 };

    Tensor2D N = Fastor::einsum<Fastor::Index<i>, Fastor::Index<j>, Fastor::OIndex<i,j>>(normal, normal);
    
    Tensor2D T = I - N;

    Tensor4D C_M_aibj = voigtToStiffness(C_M_voigt_full);
    Tensor4D C_I_aibj = voigtToStiffness(C_I_voigt_full);
    Tensor4D C_0_aibj = voigtToStiffness(C_0_voigt_full);
      
    auto [Z, H_inv, H_inv_nF, Yn_H_inv_Fn] = calculate_material_matrices(normal, I, N, T, C_0_aibj, C_M_aibj, C_I_aibj);
    
    return {Z, H_inv, H_inv_nF, Yn_H_inv_Fn}; 
}


std::tuple<Eigen::Matrix<double,3,3>, Eigen::Matrix<double,9,9>, double, double, double>calculate_unitcompliance_interface(
                                                                                           const Tensor1D& normal,
                                                                                           const double& E_M,
                                                                                           const double& nu_M,
                                                                                           const double& E_I,
                                                                                           const double& nu_I,
                                                                                           const double& E_0,
                                                                                           const double& nu_0)
{   using namespace Marmot::ContinuumMechanics::Elasticity::Isotropic;
    //Tensor2D I = {{1.0,0.0,0.0},
    //              {0.0,1.0,0.0},
    //              {0.0,0.0,1.0}
    //             };

    Tensor2D N = Fastor::einsum<Fastor::Index<i>, Fastor::Index<j>, Fastor::OIndex<i,j>>(normal, normal);

    double E_bar = E_M+E_I-2.*E_0;
    double H_bar = 2.*1./E_0 - 1./E_M -1/E_I;
    
    double nu_bar = nu_0;  
    Eigen::Matrix<double,6,6> unitC_bar_voigt_full =  stiffnessTensor(1, nu_bar);
    Tensor4D unitC_bar_tensor = voigtToStiffness(unitC_bar_voigt_full);

    Tensor2D unitQ_bar_tensor = Fastor::einsum<Fastor::Index<a,i,b,j>, Fastor::Index<i,j>, Fastor::OIndex<a,b>>(unitC_bar_tensor, N);
    Eigen::Map<Eigen::Matrix<double,3,3, Eigen::RowMajor>> unitQ_bar_voigt_full(unitQ_bar_tensor.data());

    Eigen::Matrix<double,3,3> I = Eigen::Matrix<double,3,3>::Identity();
    Eigen::Matrix<double, 3, 3> unitG_bar_voigt_full = unitQ_bar_voigt_full.fullPivLu().solve( I );
    Fastor::TensorMap<double,3,3> unitG_bar_tensor(unitG_bar_voigt_full.data());

    //Tensor2D unitG_bar_tensor = compute_inv(I, unitQ_bar_tensor );

    //static bool printed = false;
    //if(!printed)
  //{ 
    //std::cout<<"unitC_bar_tensor\n"<<unitC_bar_tensor<<"\n";
    //std::cout<<"normal:\n"<<normal<<"\n";
    //std::cout<<"N:\n"<<N<<"\n";
    //std::cout<<"unitQ_tensor:\n"<<unitQ_bar_voigt_full<<"\n";
    //std::cout<<"unitG_tensor:\n"<<unitG_bar_voigt_full<<"\n";    
    //printed = true;
  //}
    Tensor4D unitZ_bar_tensor = unitC_bar_tensor;
    Tensor4D unitA_bar_tensor = Fastor::einsum<Fastor::Index<a,b>, Fastor::Index<i,j>, Fastor::OIndex<a,i,b,j>>(unitG_bar_tensor,N); 
    unitZ_bar_tensor -= Fastor::einsum<Fastor::Index<a,i,m,n>, Fastor::Index<m,n,k,l>, Fastor::Index<k,l,b,j>, Fastor::OIndex<a,i,b,j>>(unitC_bar_tensor,unitA_bar_tensor,unitC_bar_tensor); 
    
    Eigen::Matrix<double, 9, 9> II = Eigen::Matrix<double, 9, 9>::Identity();
    Eigen::Map<Eigen::Matrix<double,9,9, Eigen::RowMajor>> unitZ_bar_voigt_full(unitZ_bar_tensor.data());
  
    Eigen::Matrix<double, 9, 9> unitZ_bar_inv_voigt_full = unitZ_bar_voigt_full.fullPivLu().solve( II );

    //Fastor::TensorMap<double,3,3,3,3> unitZ_bar_inv_tensor(unitZ_bar_inv_voigt_full.data());

    return  {unitG_bar_voigt_full, unitZ_bar_inv_voigt_full, E_bar, H_bar, nu_bar}; 
}

std::tuple<Tensor2D, Tensor4D>calculate_effective_properties(const double& barC_Ju,
                                                             const double& barC_Js, 
                                                             const Tensor1D& normal,
                                                             const double& nu_bar
                                                             )
{   using namespace Marmot::ContinuumMechanics::Elasticity::Isotropic;
    Tensor2D I = {{1.0,0.0,0.0},
                  {0.0,1.0,0.0},
                  {0.0,0.0,1.0}
                 };

    Tensor2D N = Fastor::einsum<Fastor::Index<i>, Fastor::Index<j>, Fastor::OIndex<i,j>>(normal, normal);

    Eigen::Matrix<double, 6, 6> C_Ju_full = stiffnessTensor(1./barC_Ju, nu_bar);                   
    
    Eigen::Matrix<double, 6, 6> C_Js_full = stiffnessTensor(1./barC_Js, nu_bar);

    Tensor4D C_Ju_tensor = voigtToStiffness(C_Ju_full);
    Tensor4D C_Js_tensor = voigtToStiffness(C_Js_full);

    Tensor2D H_inv_Ju_tensor = Fastor::einsum<Fastor::Index<a,i,b,j>, Fastor::Index<i,j>, Fastor::OIndex<a,b>>(C_Ju_tensor, N);
    
    Tensor2D H_inv_Js_tensor = Fastor::einsum<Fastor::Index<a,i,b,j>, Fastor::Index<i,j>, Fastor::OIndex<a,b>>(C_Js_tensor, N);
    Tensor2D G_Js_tensor = compute_inv(I, H_inv_Js_tensor);
    
    //Tensor4D A = Fastor::einsum<Fastor::Index<a,b>,Fastor::Index<i,j>,Fastor::OIndex<a,i,b,j>>(G, N );
    //Tensor4D LA = Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(L,A);
    //Tensor4D LAL = Fastor::einsum<Fastor::Index<a,i,m,n>,Fastor::Index<m,n,b,j>,Fastor::OIndex<a,i,b,j>>(LA, L );
    //Tensor4D B = L - LAL;    
    
    Tensor4D Z_Js_tensor = C_Js_tensor;
    Tensor4D A_Js_tensor = Fastor::einsum<Fastor::Index<a,b>, Fastor::Index<i,j>, Fastor::OIndex<a,i,b,j>>(G_Js_tensor,N); 
    Z_Js_tensor -= Fastor::einsum<Fastor::Index<a,i,m,n>, Fastor::Index<m,n,k,l>, Fastor::Index<k,l,b,j>, Fastor::OIndex<a,i,b,j>>(C_Js_tensor,A_Js_tensor,C_Js_tensor);
    //assert_Q_ij_G_ij(H_inv_Js_tensor, normal, 1./barC_Js);
    //assert_Z_ijkl(Z_Js_tensor, normal, 1./barC_Js);
    
    return  {H_inv_Ju_tensor, Z_Js_tensor}; 
}
