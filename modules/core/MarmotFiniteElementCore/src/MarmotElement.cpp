#include "Marmot/MarmotElement.h"

MarmotElement::~MarmotElement() {}

void MarmotElement::assignProperty( const ElementProperties& property ) {}

void MarmotElement::assignProperty( const MarmotMaterialSection& property ) {}

void MarmotElement::assignProperty( const std::string& propertyName, const double* properties ) {}

std::vector< std::string > MarmotElement::getPropertyNames() const
{
  return {};
}
