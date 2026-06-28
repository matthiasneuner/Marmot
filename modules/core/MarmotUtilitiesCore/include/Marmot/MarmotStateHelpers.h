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
#include "Fastor/Fastor.h"
#include "Marmot/MarmotUtils.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @struct StateMapper
 * @brief Template struct to map raw double pointers and sizes to specific views.
 * Specializations of this struct should provide a static `map` method that
 * takes a `double*` and a `std::size_t` size, returning the desired view type.
 */
template < class View, int... Args >
struct StateMapper;

/// \cond DOXYGEN_SKIP
// Template specialisations of StateMapper are hidden from Doxygen because
// Sphinx's C++ domain parser cannot handle deeply nested template arguments
// (e.g. StateMapper< Eigen::Map< Eigen::MatrixXd > >) and would emit
// "Invalid C++ declaration" warnings for every specialisation.

/**
 * @struct StateMapper<double>
 * @brief Specialization of StateMapper for single double values.
 * This allows mapping a raw double pointer and size to a single double reference.
 */
template <>
struct StateMapper< double& > {
  static double& map( double* ptr, std::size_t n )
  {
    if ( n != 1 )
      throw std::runtime_error( "Size mismatch for double." );
    return *ptr;
  }
};

/**
 * @struct StateMapper<const double>
 * @brief Specialization of StateMapper for single const double values.
 * This allows mapping a raw const double pointer and size to a single const double reference.
 */
template <>
struct StateMapper< const double& > {
  static const double& map( const double* ptr, std::size_t n )
  {
    if ( n != 1 )
      throw std::runtime_error( "Size mismatch for double." );
    return *ptr;
  }
};
/**
 * @struct StateMapper<Fastor::Tensor<double,3,3>>
 * @brief Specialization of StateMapper for Fastor::Tensor<double,3,3>.
 * This allows mapping a raw double pointer and size to a Fastor 3x3 tensor.
 */
template <>
struct StateMapper< Fastor::TensorMap< double, 3, 3 > > {
  static Fastor::TensorMap< double, 3, 3 > map( double* ptr, std::size_t n )
  {
    if ( n != 9 )
      throw std::runtime_error( "Size mismatch for Fastor::Tensor<double,3,3>." );
    return Fastor::TensorMap< double, 3, 3 >( ptr );
  }
};

/**
 * @struct StateMapper<Eigen::Map<Eigen::MatrixXd>>
 * @brief Specialization of StateMapper for Eigen::Map<Eigen::MatrixXd>.
 * This allows mapping a raw double pointer and size to an Eigen dynamic matrix map.
 */
template <>
struct StateMapper< Eigen::Map< Eigen::MatrixXd > > {
  template < class... Args >
  static Eigen::Map< Eigen::MatrixXd > map( double* ptr, std::size_t n, int Rows, int Cols )
  {
    // read size form Args
    if ( int( n ) != Rows * Cols )
      throw std::runtime_error( "Size mismatch for Eigen::Map." );
    return Eigen::Map< Eigen::MatrixXd >( ptr, Rows, n / Rows );
  }
};

/**
 * @struct StateMapper<Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols>>>
 * @brief Specialization of StateMapper for fixed-size Eigen::Map types.
 */
template < typename Scalar, int Rows, int Cols >
struct StateMapper< Eigen::Map< Eigen::Matrix< Scalar, Rows, Cols > > > {
  template < class... Args >
  static Eigen::Map< Eigen::Matrix< Scalar, Rows, Cols > > map( double* ptr, std::size_t n, Args&&... )
  {
    // Check if the expected total size matches the provided size 'n'
    if ( int( n ) != Rows * Cols )
      throw std::runtime_error( "Size mismatch for fixed-size Eigen::Map." );

    // Map the raw pointer to the Eigen fixed-size matrix map
    return Eigen::Map< Eigen::Matrix< Scalar, Rows, Cols > >( ptr, Rows, Cols );
  }
};
/// \endcond

template < std::size_t N >
struct FixedString {
  char value[N];
  constexpr FixedString( const char ( &str )[N] ) { std::copy_n( str, N, value ); }
};

template < typename View, std::size_t Size, FixedString Name >
struct StateVar {
  using view_type                   = View;
  static constexpr std::size_t size = Size;
  static constexpr const char* name = Name.value;
};

/**
 * @class MarmotStateLayoutDynamic
 * @brief Runtime-defined layout for a flat array of named `double` state variables.
 *
 * Variables are registered by name and size via add(), then the layout is locked
 * with finalize(), which assigns contiguous offsets. After finalization the
 * various `get*` accessors can be used to reach individual variables inside any
 * conforming state vector.
 */
class MarmotStateLayoutDynamic {
public:
  /**
   * @brief Describes a single named state variable registered in the layout.
   */
  struct VarInfo {
    std::string name;   ///< Unique name of the state variable.
    std::size_t size;   ///< Number of `double` values occupied by this variable.
    std::size_t offset; ///< Byte-offset (in units of `double`) from the beginning of the state vector.
  };

  /**
   * @brief Register a new named state variable and return its index.
   *
   * Must be called before finalize(). Adding a variable with an already-registered name
   * or calling this method after finalize() throws @c std::runtime_error.
   *
   * @param name  Unique name for the state variable.
   * @param size  Number of `double` values required by the variable.
   * @return      The zero-based index of the registered variable.
   */
  std::size_t add( std::string name, std::size_t size )
  {
    if ( finalized )
      throw std::runtime_error( "Cannot add variables after finalize()." );

    if ( name_to_index.contains( name ) )
      throw std::runtime_error( "Duplicate state variable: " + name );

    std::size_t index   = variables.size();
    name_to_index[name] = index;
    variables.push_back( { std::move( name ), size, 0 } );
    return index;
  }

  /**
   * @brief Compute and lock the offsets of all registered variables.
   *
   * Iterates over the registered variables in registration order, assigns each a
   * contiguous offset, and sets the total size. After this call no further variables
   * may be added. Calling finalize() a second time throws @c std::runtime_error.
   */
  void finalize()
  {
    if ( finalized )
      throw std::runtime_error( "State layout already finalized." );
    std::size_t offset = 0;
    for ( auto& v : variables ) {
      v.offset = offset;
      offset += v.size;
    }
    total_sz  = offset;
    finalized = true;
  }

  /**
   * @brief Return the index of a registered variable by name.
   *
   * @param name  Name of the variable to look up.
   * @return      The zero-based index of the registered variable.
   * @throws std::runtime_error if @p name is not registered.
   */
  std::size_t getIndex( const std::string& name ) const
  {
    auto it = name_to_index.find( name );
    if ( it == name_to_index.end() )
      throw std::runtime_error( "Unknown state variable: " + name );
    return it->second;
  }

  /**
   * @brief Return the @ref VarInfo record for a registered variable.
   *
   * @param name  Name of the variable to look up.
   * @return      Const reference to the corresponding @ref VarInfo.
   * @throws std::runtime_error if @p name is not registered.
   */
  const VarInfo& getInfo( const std::string& name ) const
  {
    auto it = name_to_index.find( name );
    if ( it == name_to_index.end() )
      throw std::runtime_error( "Unknown state variable: " + name );
    return variables[it->second];
  }

  /**
   * @brief Return the {offset, size} pair for a registered variable by name.
   *
   * @param name  Name of the variable.
   * @return      A pair `{offset, size}` (both in units of `double`).
   * @throws std::runtime_error if @p name is not registered.
   */
  std::pair< std::size_t, std::size_t > get( const std::string& name ) const
  {
    const auto& v = getInfo( name );
    return { v.offset, v.size };
  }

  /**
   * @brief Return the {offset, size} pair for a registered variable by index.
   *
   * @param index  Index of the variable.
   * @return       A pair `{offset, size}` (both in units of `double`).
   * @throws std::out_of_range if @p index is invalid.
   */
  std::pair< std::size_t, std::size_t > get( std::size_t index ) const
  {
    if ( index >= variables.size() )
      throw std::out_of_range( "State variable index out of range." );
    const auto& v = variables[index];
    return { v.offset, v.size };
  }

  /**
   * @brief Return a raw pointer to a variable inside a state vector by name.
   *
   * @param base  Pointer to the beginning of the state vector.
   * @param name  Name of the variable.
   * @return      `base + offset` for the named variable.
   * @throws std::runtime_error if @p name is not registered.
   */
  double* getPtr( double* base, const std::string& name ) const
  {
    const auto& v = getInfo( name );
    return base + v.offset;
  }

  /**
   * @brief Return a raw pointer to a variable inside a state vector by index.
   *
   * @param base   Pointer to the beginning of the state vector.
   * @param index  Index of the variable.
   * @return       `base + offset` for the indexed variable.
   * @throws std::out_of_range if @p index is invalid.
   */
  double* getPtr( double* base, std::size_t index ) const
  {
    if ( index >= variables.size() )
      throw std::out_of_range( "State variable index out of range." );
    return base + variables[index].offset;
  }

  /**
   * @brief Return a const raw pointer to a variable inside a state vector by index.
   *
   * @param base   Pointer to the beginning of the state vector.
   * @param index  Index of the variable.
   * @return       `base + offset` for the indexed variable.
   * @throws std::out_of_range if @p index is invalid.
   */
  const double* getPtr( const double* base, std::size_t index ) const
  {
    if ( index >= variables.size() )
      throw std::out_of_range( "State variable index out of range." );
    return base + variables[index].offset;
  }

  /**
   * @brief Return a `std::span<double>` view of a variable inside a state vector by name.
   *
   * @param base  Pointer to the beginning of the state vector.
   * @param name  Name of the variable.
   * @return      A span covering `[base + offset, base + offset + size)`.
   * @throws std::runtime_error if @p name is not registered.
   */
  std::span< double > getSpan( double* base, const std::string& name ) const
  {
    const auto& v = getInfo( name );
    return { base + v.offset, v.size };
  }

  /**
   * @brief Return a `std::span<double>` view of a variable inside a state vector by index.
   *
   * @param base   Pointer to the beginning of the state vector.
   * @param index  Index of the variable.
   * @return       A span covering the variable.
   * @throws std::out_of_range if @p index is invalid.
   */
  std::span< double > getSpan( double* base, std::size_t index ) const
  {
    if ( index >= variables.size() )
      throw std::out_of_range( "State variable index out of range." );
    const auto& v = variables[index];
    return { base + v.offset, v.size };
  }

  /**
   * @brief Return a @ref StateView for a variable inside a state vector by name.
   *
   * @param base  Pointer to the beginning of the state vector.
   * @param name  Name of the variable.
   * @return      A @ref StateView with `stateLocation = base + offset` and `stateSize = size`.
   * @throws std::runtime_error if @p name is not registered.
   */
  StateView getStateView( double* base, const std::string& name ) const
  {
    const auto& v = getInfo( name );
    return StateView{ base + v.offset, static_cast< int >( v.size ) };
  }

  /**
   * @brief Return a @ref StateView for a variable inside a state vector by index.
   *
   * @param base   Pointer to the beginning of the state vector.
   * @param index  Index of the variable.
   * @return       A @ref StateView with offset and size.
   * @throws std::out_of_range if @p index is invalid.
   */
  StateView getStateView( double* base, std::size_t index ) const
  {
    if ( index >= variables.size() )
      throw std::out_of_range( "State variable index out of range." );
    const auto& v = variables[index];
    return StateView{ base + v.offset, static_cast< int >( v.size ) };
  }

  /**
   * @brief Map a variable inside a state vector to a typed view via `StateMapper` by name.
   *
   * Delegates to `StateMapper<View>::map(base + offset, size, args...)`.
   * The layout must have been finalized before calling this method.
   *
   * @tparam View   Target view type (must have a matching `StateMapper` specialization).
   * @tparam Args   Additional constructor arguments forwarded to `StateMapper::map`.
   * @param base    Pointer to the beginning of the state vector.
   * @param name    Name of the variable.
   * @param args    Extra arguments forwarded to `StateMapper::map`.
   * @return        The mapped view of the requested variable.
   * @throws std::runtime_error if the layout is not finalized or @p name is not registered.
   */
  template < class View, class... Args >
  View getAs( double* base, const std::string& name, Args&&... args ) const
  {
    if ( !finalized )
      throw std::runtime_error( "State layout not finalized." );

    const auto& v = getInfo( name );
    return StateMapper< View >::map( base + v.offset, v.size, std::forward< Args >( args )... );
  }

  /**
   * @brief Map a variable inside a state vector to a typed view via `StateMapper` by index.
   *
   * @tparam View   Target view type (must have a matching `StateMapper` specialization).
   * @tparam Args   Additional constructor arguments forwarded to `StateMapper::map`.
   * @param base    Pointer to the beginning of the state vector.
   * @param index   Index of the variable.
   * @param args    Extra arguments forwarded to `StateMapper::map`.
   * @return        The mapped view of the requested variable.
   * @throws std::runtime_error if the layout is not finalized.
   * @throws std::out_of_range if @p index is invalid.
   */
  template < class View, class... Args >
  View getAs( double* base, std::size_t index, Args&&... args ) const
  {
    if ( !finalized )
      throw std::runtime_error( "State layout not finalized." );

    if ( index >= variables.size() )
      throw std::out_of_range( "State variable index out of range." );

    const auto& v = variables[index];
    return StateMapper< View >::map( base + v.offset, v.size, std::forward< Args >( args )... );
  }

  /**
   * @brief Const version of @ref getAs by name.
   * @tparam View   Target view type (must have a matching `StateMapper` specialization).
   * @tparam Args   Additional constructor arguments forwarded to `StateMapper::map`.
   * @param base    Pointer to the beginning of the state vector.
   * @param name    Name of the variable.
   * @param args    Extra arguments forwarded to `StateMapper::map`.
   * @return        The mapped view of the requested variable.
   * @throws std::runtime_error if the layout is not finalized or @p name is not registered.
   */
  template < class View, class... Args >
  View getAs( const double* base, const std::string& name, Args&&... args ) const
  {
    if ( !finalized )
      throw std::runtime_error( "State layout not finalized." );

    const auto& v = getInfo( name );
    return StateMapper< View >::map( base + v.offset, v.size, std::forward< Args >( args )... );
  }

  /**
   * @brief Const version of index-based @ref getAs.
   */
  template < class View, class... Args >
  View getAs( const double* base, std::size_t index, Args&&... args ) const
  {
    if ( !finalized )
      throw std::runtime_error( "State layout not finalized." );

    if ( index >= variables.size() )
      throw std::out_of_range( "State variable index out of range." );

    const auto& v = variables[index];
    return StateMapper< View >::map( base + v.offset, v.size, std::forward< Args >( args )... );
  }

  /**
   * @brief Return the total size (in units of `double`) of the state layout.
   * This is the size of the state vector required to hold all registered variables.
   */
  int totalSize() const { return total_sz; }

  /**
   * @brief Check if the layout has been finalized.
   * @return `true` if finalize() has been called, `false` otherwise.
   */
  bool isFinalized() const { return finalized; }

private:
  std::vector< VarInfo >                         variables;
  std::unordered_map< std::string, std::size_t > name_to_index;

  bool finalized = false;
  int  total_sz  = 0;
};

template < std::size_t I, typename Tuple >
struct OffsetOf;

template < typename Tuple >
struct OffsetOf< 0, Tuple > {
  static constexpr std::size_t value = 0;
};

template < std::size_t I, typename Tuple >
struct OffsetOf {
  static constexpr std::size_t value = std::tuple_element_t< I - 1, Tuple >::size + OffsetOf< I - 1, Tuple >::value;
};

template < typename... Vars >
class CompileTimeStateLayout {
  using tuple_type                      = std::tuple< Vars... >;
  static constexpr std::size_t num_vars = sizeof...( Vars );

public:
  static constexpr std::size_t total_size = OffsetOf< num_vars, tuple_type >::value;

  template < std::size_t I >
  static constexpr std::size_t offset = OffsetOf< I, tuple_type >::value;

  // Registers all variables to the dynamic layout for visualization/solvers
  static void registerLayout( MarmotStateLayoutDynamic& dynamicLayout )
  {
    auto reg = [&]< std::size_t... Is >( std::index_sequence< Is... > ) {
      ( dynamicLayout.add( std::tuple_element_t< Is, tuple_type >::name, std::tuple_element_t< Is, tuple_type >::size ),
        ... );
    };
    reg( std::make_index_sequence< num_vars >{} );
    dynamicLayout.finalize();
  }

  // High-performance typed compile-time getter for non-const base pointer
  template < std::size_t I, typename BaseType, typename = std::enable_if_t< !std::is_const_v< BaseType > > >
  static decltype( auto ) get( BaseType* base )
  {
    using Var = std::tuple_element_t< I, tuple_type >;
    return StateMapper< typename Var::view_type >::map( base + offset< I >, Var::size );
  }

  // High-performance typed compile-time getter for const base pointer
  template < std::size_t I >
  static decltype( auto ) get( const double* base )
  {
    using Var       = std::tuple_element_t< I, tuple_type >;
    using view_type = typename Var::view_type;
    if constexpr ( std::is_same_v< view_type, double& > ) {
      return StateMapper< const double& >::map( base + offset< I >, Var::size );
    }
    else {
      return StateMapper< view_type >::map( const_cast< double* >( base ) + offset< I >, Var::size );
    }
  }
};
