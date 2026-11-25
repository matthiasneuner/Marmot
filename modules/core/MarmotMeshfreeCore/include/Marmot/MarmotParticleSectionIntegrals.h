/* ---------------------------------------------------------------------
 *                                       _
 *  _ __ ___   __ _ _ __ _ __ ___   ___ | |_
 * | '_ ` _ \ / _` | '__| '_ ` _ \ / _ \| __|
 * | | | | | | (_| | |  | | | | | | (_) | |_
 * |_| |_| |_|\__,_|_|  |_| |_| |_|\___/ \__|
 *
 * Unit of Strength of Materials and Structural Analysis
 * University of Innsbruck
 * 2020 - today
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
#pragma once

#include "Marmot/MarmotConstants.h"
#include "Marmot/MarmotMeshfreeApproximation.h"
#include "Marmot/MarmotMeshfreeKernelFunction.h"
#include "Marmot/MarmotMeshfreeQuadHexCell.h"
#include "Marmot/MarmotParticle.h"
#include "Marmot/MarmotParticleStiffness.h"
#include "Marmot/MarmotUtils.h"
#include <Eigen/Core>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

/**
 * @file
 * @brief The integrals over the smoothing domain of a particle in its section: its volume, the smoothed gradients of
 * the shape functions and its second moments, one implementation for the SQCNI, SNNI, NSNI and SDI particles of both
 * families; and the hoop terms of the axisymmetric weak form. The axisymmetric derivation is on the page
 * "Axisymmetric Particles" of the documentation.
 */

namespace Marmot::Discretizations::Meshfree {

  /**
   * @brief The integrals over the smoothing domain of a particle in its section: its volume, the smoothed gradients of
   * the shape functions (stabilized conforming nodal integration, Chen et al., 2001) and its second moments.
   *
   * @details The smoothed gradient is the average of the gradient over the smoothing domain @f$ \Omega_s @f$, turned
   * into a boundary integral by the divergence theorem and integrated with a rule on each face @f$ f @f$
   * (faceQuadraturePoints()). In the Cartesian section (plane strain, 3D) the rule is one point per face, the face
   * center @f$ \boldsymbol{Y}_f @f$,
   * @f[
   *   \tilde N_{A,i} = \frac{1}{|\Omega_s|}\int_{\Omega_s} \frac{\partial N_A}{\partial Y_i}\,dV
   *   = \frac{1}{|\Omega_s|}\oint_{\partial\Omega_s} N_A\,n_i\,dA
   *   \approx \frac{1}{|\Omega_s|} \sum_f N_A(\boldsymbol{Y}_f)\, n_{f,i}\, dA_f ,
   * @f]
   * with the coordinates @f$ \boldsymbol{Y} @f$ of the intermediate configuration, the volume @f$ |\Omega_s| @f$
   * (volume()) and the area vector @f$ \boldsymbol{n}_f\,dA_f @f$ of face @f$ f @f$
   * (LagrangeCell::boundarySurfaceVector()). Here the face rule is the only approximation, and only the values of the
   * shape functions at the face points are needed. NSNI smooths the second derivatives alike, from the (unsmoothed)
   * gradients at the same points,
   * @f[
   *   \hat N_{A,IJ} = \frac{1}{|\Omega_s|} \sum_f \frac{\partial N_A}{\partial Y_I}(\boldsymbol{Y}_f)\,
   *   n_{f,J}\, dA_f ,
   * @f]
   * stored unsymmetrized as `d2N_dYdY[J](I, A)`; NSNI::secondGradientOfNode() symmetrizes them.
   *
   * In the axisymmetric section, @f$ \Omega_s @f$ is the ring swept by the smoothing domain of the meridian half plane
   * (@f$ Y_0 = r @f$), and every integral carries the ring weight @f$ 2\pi r @f$ (weight()): @f$ |\Omega_s| = 2\pi
   * A_s \bar r @f$, with the area @f$ A_s @f$ and the radius @f$ \bar r @f$ of its area centroid (the particle center
   * @f$ \bar{\boldsymbol{Y}} @f$). The divergence theorem of the weighted gradient gains an area term, which is
   * integrated with the centroid; the faces are integrated with two Gauss points @f$ \boldsymbol{Y}_g @f$ each, which
   * share the area vector @f$ \boldsymbol{n}_f\,dA_f @f$ of the face (in the half plane) equally:
   * @f[
   *   \tilde N_{A,i} = \frac{1}{|\Omega_s|} \sum_f \sum_{g=1}^{2} 2\pi r_g\, N_A(\boldsymbol{Y}_g)\,
   *   \frac{n_{f,i}\, dA_f}{2} - \delta_{ir}\, \frac{N_A(\bar{\boldsymbol{Y}})}{\bar r} .
   * @f]
   * This is the exact ring average of the gradient for every linear field, and it satisfies the integration
   * constraint together with the hoop operator @f$ N_A / \bar r @f$ (Hoop). The second derivatives of NSNI are
   * smoothed alike,
   * @f[
   *   \hat N_{A,IJ} = \frac{1}{|\Omega_s|} \sum_f \sum_{g=1}^{2} 2\pi \Big( r_g\, \frac{\partial N_A}{\partial
   *   Y_I}(\boldsymbol{Y}_g)\, n_{f,J} - \delta_{Jr}\, N_A(\boldsymbol{Y}_g)\, n_{f,I} \Big) \frac{dA_f}{2} ,
   * @f]
   * and the second moments are those of the ring about the area centroid, @f$ 2\pi \int \boldsymbol{\xi} \otimes
   * \boldsymbol{\xi}\, r\, dA @f$.
   *
   * The particles take these integrals, and their evaluation points (faceQuadraturePoints()), from here and not from
   * the cell: the face rule of the smoothing is defined in this class only. *Section* is meant as for finite elements
   * (plane strain, 3D, axisymmetric): the form of the integrals depends on it; in the Cartesian section the integrals
   * carry the weight 1. The class holds static functions only, so that a particle selects its integrals with its own
   * template parameters.
   *
   * @tparam section   The section.
   * @tparam nDim      The spatial dimension; 2 for the axisymmetric section.
   * @tparam nVertices The number of vertices of the cell (4: Quad, 8: Hexa).
   */
  template < Section section, int nDim, int nVertices >
  class ParticleSectionIntegrals {
    static_assert( section == Section::Cartesian || nDim == 2,
                   "an axisymmetric particle lives in the meridian half plane: nDim = 2" );

  public:
    using Cell = LagrangeCell< nDim, nVertices >;  ///< a smoothing domain (a Lagrange cell)
    using Vec  = Eigen::Matrix< double, nDim, 1 >; ///< a point or a vector

    /// @brief The section is axisymmetric.
    static constexpr bool isAxisymmetric = section == Section::Axisymmetric;

    /// @brief The number of quadrature points per face of a smoothing domain: one, the face center (Cartesian), or two
    /// Gauss points (axisymmetric).
    static constexpr int nPointsPerFace = isAxisymmetric ? 2 : 1;

    /// @brief The quadrature points of one face, nDim x nPointsPerFace.
    using FacePoints = Eigen::Matrix< double, nDim, nPointsPerFace >;

    /**
     * @brief The volume of a smoothing domain.
     * @param[in] cell The smoothing domain.
     * @return Its volume @f$ |\Omega_s| @f$ (area in plane strain), or its ring volume @f$ 2\pi A_s \bar r @f$
     * (Pappus).
     */
    static double volume( const Cell& cell )
    {
      if constexpr ( isAxisymmetric )
        return 2.0 * Marmot::Math::Constants::Pi * cell.volume() * cell.centroid()( 0 );
      else
        return cell.volume();
    }

    /**
     * @brief The weight of the section at a point, the factor of the area element of a face.
     * @param[in] Y The point.
     * @return 1, or the ring @f$ 2\pi r @f$.
     */
    static double weight( const Vec& Y )
    {
      if constexpr ( isAxisymmetric )
        return 2.0 * Marmot::Math::Constants::Pi * Y( 0 );
      else
        return 1.0;
    }

    /**
     * @brief The number of face quadrature points of a smoothing domain.
     * @param[in] cell The smoothing domain.
     * @return The number of face quadrature points, `faceQuadraturePoints( cell ).cols()`.
     */
    static int numberOfFaceQuadraturePoints( const Cell& cell ) { return cell.getNumberOfFaces() * nPointsPerFace; }

    /**
     * @brief The quadrature points of one face of a smoothing domain: its center, or its two Gauss points.
     * @param[in] cell The smoothing domain.
     * @param[in] faceID The face (1-based).
     * @return nDim x nPointsPerFace: the face center (Cartesian), or the Gauss points @f$ \boldsymbol{Y}_f \mp
     * \boldsymbol{e}_f / (2\sqrt{3}) @f$ of the straight edge @f$ \boldsymbol{e}_f @f$ (axisymmetric).
     */
    static FacePoints faceQuadraturePoints( const Cell& cell, int faceID )
    {
      if constexpr ( isAxisymmetric ) {
        // the edge from the surface vector (e_1, -e_0) of a counter-clockwise quadrilateral
        const Vec  n_dA = cell.boundarySurfaceVector( faceID );
        const Vec  edge( -n_dA( 1 ), n_dA( 0 ) );
        const Vec  center = cell.getFaceCenterCoordinates( faceID );
        FacePoints points;
        for ( int g = 0; g < nPointsPerFace; g++ )
          points.col( g ) = center + ( g == 0 ? -1.0 : 1.0 ) / ( 2.0 * std::sqrt( 3.0 ) ) * edge;
        return points;
      }
      else
        return cell.getFaceCenterCoordinates( faceID );
    }

    /**
     * @brief The quadrature points of all faces of a smoothing domain.
     * @param[in] cell The smoothing domain.
     * @return nDim x numberOfFaceQuadraturePoints(), face by face.
     */
    static Eigen::Matrix< double, nDim, Eigen::Dynamic > faceQuadraturePoints( const Cell& cell )
    {
      Eigen::Matrix< double, nDim, Eigen::Dynamic > points( nDim, numberOfFaceQuadraturePoints( cell ) );
      for ( int f = 0; f < cell.getNumberOfFaces(); f++ )
        points.middleCols( f * nPointsPerFace, nPointsPerFace ) = faceQuadraturePoints( cell, f + 1 );
      return points;
    }

    /**
     * @brief The smoothed gradients (and second derivatives) of the shape functions over a smoothing domain, see the
     * class description. The points of a face share its area vector equally.
     * @param[in] smoothingDomain The smoothing domain; its area centroid is the particle center.
     * @param[in] approximation The meshfree approximation.
     * @param[in] kernelFunctions The kernel functions of the nodes.
     * @param[in] N The shape functions at the particle center, 1 x nNodes: the area term of the ring (axisymmetric);
     * the Cartesian smoothing uses only its size (the number of nodes).
     * @param[out] dN_dY The smoothed gradients, nDim x nNodes.
     * @param[out] d2N_dYdY The smoothed second derivatives, `d2N_dYdY[J](I, A)`; nullptr: first derivatives only
     * (SQCNI, SNNI, SDI); NSNI passes its array.
     */
    static void smoothShapeFunctionGradients( const Cell&                                 smoothingDomain,
                                              const Approximation&                        approximation,
                                              const std::vector< const KernelFunction* >& kernelFunctions,
                                              const Eigen::MatrixXd&                      N,
                                              Eigen::MatrixXd&                            dN_dY,
                                              std::array< Eigen::MatrixXd, nDim >*        d2N_dYdY );

    /**
     * @brief The second moments of a domain about its area centroid in the section, @f$ \int \boldsymbol{\xi} \otimes
     * \boldsymbol{\xi}\, dV @f$.
     * @param[in] cell The domain.
     * @return The second moments of the cell, or of its ring (a 3 x 3 Gauss rule, exact for a quadrilateral).
     */
    static Eigen::Matrix< double, nDim, nDim > secondMoments( const Cell& cell );

    /**
     * @brief Refuse a domain the section cannot integrate: an axisymmetric particle with a vertex at @f$ r < 0 @f$
     * (beyond the round-off of a mesher, relative to the diagonal of the bounding box) or its centroid at
     * @f$ r \le 0 @f$, where the hoop strain @f$ u_r / r @f$ is undefined and the ring weight not positive. Nothing to
     * check for the Cartesian section.
     * @param[in] cell The undeformed domain.
     * @param[in] particleLabel The label of the particle, for the message.
     * @throws std::invalid_argument for such a domain.
     */
    static void checkDomain( const Cell& cell, int particleLabel );
  };

  template < Section section, int nDim, int nVertices >
  void ParticleSectionIntegrals< section, nDim, nVertices >::smoothShapeFunctionGradients(
    const Cell&                                 smoothingDomain,
    const Approximation&                        approximation,
    const std::vector< const KernelFunction* >& kernelFunctions,
    const Eigen::MatrixXd&                      N,
    Eigen::MatrixXd&                            dN_dY,
    std::array< Eigen::MatrixXd, nDim >*        d2N_dYdY )
  {
    const int nNodes = static_cast< int >( N.cols() );
    dN_dY            = Eigen::MatrixXd::Zero( nDim, nNodes );
    if ( d2N_dYdY )
      for ( int J = 0; J < nDim; J++ )
        ( *d2N_dYdY )[J] = Eigen::MatrixXd::Zero( nDim, nNodes );

    Eigen::MatrixXd NBoundary( 1, nNodes );
    Eigen::MatrixXd dN_dY_Boundary = Eigen::MatrixXd::Zero( nDim, nNodes );

    for ( int f = 0; f < smoothingDomain.getNumberOfFaces(); f++ ) {
      const FacePoints points = faceQuadraturePoints( smoothingDomain, f + 1 );
      const Vec        n_dA   = smoothingDomain.boundarySurfaceVector( f + 1 ) / double( nPointsPerFace );

      for ( int g = 0; g < nPointsPerFace; g++ ) {
        const Vec Y = points.col( g );

        if ( d2N_dYdY )
          approximation.computeShapeFunctionsAndGradients( Y.data(),
                                                           kernelFunctions,
                                                           NBoundary.data(),
                                                           dN_dY_Boundary.data() );
        else
          approximation.computeShapeFunctions( Y.data(), kernelFunctions, NBoundary.data() );

        // the point's share of the face, weighted by the section (the 2 pi r of the ring)
        const Vec ring_n_dA = weight( Y ) * n_dA;
        dN_dY += ring_n_dA * NBoundary;
        if ( d2N_dYdY ) {
          for ( int J = 0; J < nDim; J++ )
            ( *d2N_dYdY )[J] += ring_n_dA( J ) * dN_dY_Boundary;
          // the area term of the ring in the second derivatives, int dN/dY_I dA = oint N n_I ds
          if constexpr ( isAxisymmetric )
            ( *d2N_dYdY )[0] -= ( 2.0 * Marmot::Math::Constants::Pi * n_dA ) * NBoundary;
        }
      }
    }

    // the area term of the ring, -delta_ir 2 pi int N dA, with the centroid rule: -2 pi A N(center)
    if constexpr ( isAxisymmetric )
      dN_dY.row( 0 ) -= ( 2.0 * Marmot::Math::Constants::Pi * smoothingDomain.volume() ) * N;

    const double V = volume( smoothingDomain );
    dN_dY /= V;
    if ( d2N_dYdY )
      for ( int J = 0; J < nDim; J++ )
        ( *d2N_dYdY )[J] /= V;
  }

  template < Section section, int nDim, int nVertices >
  Eigen::Matrix< double, nDim, nDim > ParticleSectionIntegrals< section, nDim, nVertices >::secondMoments(
    const Cell& cell )
  {
    if constexpr ( isAxisymmetric ) {
      const Vec                           C       = cell.centroid();
      const std::array< double, 3 >       points  = { -std::sqrt( 0.6 ), 0.0, std::sqrt( 0.6 ) };
      const std::array< double, 3 >       weights = { 5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0 };
      Eigen::Matrix< double, nDim, nDim > moments = Eigen::Matrix< double, nDim, nDim >::Zero();
      for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++ ) {
          const Vec    s( points[i], points[j] );
          const Vec    Y  = cell.mapToPhysical( s );
          const Vec    xi = Y - C;
          const double dV = weight( Y ) * std::abs( cell.jacobian( s ).determinant() ) * weights[i] * weights[j];
          moments += xi * xi.transpose() * dV;
        }
      return moments;
    }
    else
      return cell.secondMoments();
  }

  template < Section section, int nDim, int nVertices >
  void ParticleSectionIntegrals< section, nDim, nVertices >::checkDomain( const Cell& cell, int particleLabel )
  {
    if constexpr ( isAxisymmetric ) {
      // the round-off of a mesher, relative to the size of the particle, as for the axisymmetric elements
      constexpr double axisTolerance = 1e-12;
      const auto&      vertices      = cell.nodes();
      const double     h             = ( vertices.rowwise().maxCoeff() - vertices.rowwise().minCoeff() ).norm();
      Eigen::Index     i             = 0;
      const double     r             = vertices.row( 0 ).minCoeff( &i );
      if ( r < -axisTolerance * h )
        throw std::invalid_argument( Marmot::Utilities::MakeString()
                                     << "axisymmetric particle " << particleLabel << ": vertex " << i + 1
                                     << " lies at r = " << r
                                     << " < 0; axisymmetric particles live in the meridian half plane r >= 0." );
      const double rCenter = cell.centroid()( 0 );
      if ( rCenter <= 0.0 )
        throw std::invalid_argument( Marmot::Utilities::MakeString()
                                     << "axisymmetric particle " << particleLabel << ": its center lies at r = "
                                     << rCenter << " <= 0, where the hoop strain u_r / r is undefined." );
    }
  }

  /**
   * @brief The hoop terms of the weak form of an axisymmetric particle, shared by its formulations.
   *
   * @details The increment of the hoop stretch is @f$ \Delta F_{\theta\theta} = 1 + \Delta u_r / \bar r @f$, with the
   * radial displacement increment @f$ \Delta u_r = N_B\, \Delta q_{Br} @f$ at the particle center and its radius
   * @f$ \bar r @f$ in the intermediate configuration. The virtual work of the hoop stress adds to the radial force of
   * node @f$ A @f$ the term @f$ T_A\, \tau_{\theta\theta} / (\bar r\, \Delta F_{\theta\theta})\, V_0 @f$.
   */
  namespace Hoop {

    /**
     * @brief Refuse an increment that pulls the ring of a particle through the axis.
     * @param[in] deltaF33 The increment of the hoop stretch, @f$ \Delta F_{\theta\theta} = 1 + \Delta u_r / \bar r
     * @f$.
     * @param[in] r The radius @f$ \bar r @f$ of the center in the intermediate configuration.
     * @param[in] particleLabel The label of the particle, for the message.
     * @throws std::runtime_error if @p deltaF33 is not positive: the increment cannot be integrated, which a solver may
     * answer with a smaller increment.
     */
    inline void checkStretchIncrement( double deltaF33, double r, int particleLabel )
    {
      if ( !( deltaF33 > 0.0 ) )
        throw std::runtime_error( Marmot::Utilities::MakeString()
                                  << "axisymmetric particle " << particleLabel
                                  << ": the hoop stretch increment 1 + du_r / r = " << deltaF33 << " at r = " << r
                                  << " is not positive; the ring is pulled through the axis." );
    }

    /**
     * @brief The hoop part of the radial force of node @f$ A @f$, @f$ T_A\, \tau_{\theta\theta} / (\bar r\, \Delta
     * F_{\theta\theta})\, V_0 @f$.
     * @param[in] T_A The test function of the node at the center.
     * @param[in] r The radius @f$ \bar r @f$ of the center.
     * @param[in] deltaF33 The increment of the hoop stretch.
     * @param[in] tau33 The hoop Kirchhoff stress.
     * @param[in] V0 The undeformed ring volume of the particle.
     * @return The force.
     */
    inline double force( double T_A, double r, double deltaF33, double tau33, double V0 )
    {
      return T_A / ( r * deltaF33 ) * tau33 * V0;
    }

    /**
     * @brief Add the hoop terms of the displacement rows to the stiffness of node @f$ A @f$ (see StiffnessOfNode; the
     * displacement comes first among the dofs of a node): the hoop row through the in-plane gradient and through the
     * radial displacement increment, and every displacement row through the hoop stretch in the stress,
     * @f[
     *   \frac{\partial f_{Ar}}{\partial (\partial \Delta u_k / \partial Y_l)} \mathrel{+}= c\,
     *   \frac{\partial \tau_{\theta\theta}}{\partial \Delta F_{kl}}, \qquad
     *   \frac{\partial f_{Ar}}{\partial \Delta u_r} \mathrel{+}= \frac{c}{\bar r} \Big( \frac{\partial
     *   \tau_{\theta\theta}}{\partial \Delta F_{\theta\theta}} - \frac{\tau_{\theta\theta}}{\Delta F_{\theta\theta}}
     *   \Big), \qquad
     *   \frac{\partial f_{Aj}}{\partial \Delta u_r} \mathrel{+}= \frac{V_0}{\bar r} \frac{\partial T_A}{\partial x_i}
     *   \frac{\partial \tau_{ij}}{\partial \Delta F_{\theta\theta}},
     * @f]
     * with @f$ c = T_A V_0 / (\bar r\, \Delta F_{\theta\theta}) @f$.
     * @tparam nDofPerNode The number of dofs per node.
     * @param[in] T_A The test function of the node at the center.
     * @param[in] dT_A_dx Its spatial gradient.
     * @param[in] r The radius @f$ \bar r @f$ of the center.
     * @param[in] deltaF33 The increment of the hoop stretch.
     * @param[in] V0 The undeformed ring volume of the particle.
     * @param[in] tau33 The hoop Kirchhoff stress.
     * @param[in] dTau_dDeltaF33 @f$ \partial \tau_{ij} / \partial \Delta F_{\theta\theta} @f$, in-plane @f$ ij @f$.
     * @param[in] dTau33_dDeltaF @f$ \partial \tau_{\theta\theta} / \partial \Delta F_{kl} @f$, in-plane @f$ kl @f$.
     * @param[in] dTau33_dDeltaF33 @f$ \partial \tau_{\theta\theta} / \partial \Delta F_{\theta\theta} @f$.
     * @param[in,out] ofNode The stiffness of the node; the hoop terms are added.
     */
    template < int nDofPerNode >
    void addStiffnessOfNode( double                                T_A,
                             const Fastor::Tensor< double, 2 >&    dT_A_dx,
                             double                                r,
                             double                                deltaF33,
                             double                                V0,
                             double                                tau33,
                             const Fastor::Tensor< double, 2, 2 >& dTau_dDeltaF33,
                             const Fastor::Tensor< double, 2, 2 >& dTau33_dDeltaF,
                             double                                dTau33_dDeltaF33,
                             StiffnessOfNode< 2, nDofPerNode >&    ofNode )
    {
      const double c = T_A * V0 / ( r * deltaF33 );
      for ( int l = 0; l < 2; l++ )
        for ( int k = 0; k < 2; k++ )
          ofNode.gradientBlock( l )( 0, k ) += c * dTau33_dDeltaF( k, l );
      ofNode.df_dIncrement( 0, 0 ) += c / r * ( dTau33_dDeltaF33 - tau33 / deltaF33 );
      for ( int j = 0; j < 2; j++ ) {
        double dfj_dDeltaF33 = 0.0;
        for ( int i = 0; i < 2; i++ )
          dfj_dDeltaF33 += dT_A_dx( i ) * dTau_dDeltaF33( i, j );
        ofNode.df_dIncrement( j, 0 ) += V0 / r * dfj_dDeltaF33;
      }
    }

  } // namespace Hoop

} // namespace Marmot::Discretizations::Meshfree
