#include "Highs.h"
#include "HighsExternalDeps.h"
#include "catch.hpp"

// simple test for now
// extended tests require workflow changes to include/exclude dependencies
TEST_CASE("HighsExternalDeps", "[highs_external_deps]") {
  // tryLoad without a path — returns true only if extras are available
  bool loaded = HighsExternalDeps::tryLoad();

  std::string status = HighsExternalDeps::getLoadStatus();
  REQUIRE(!status.empty());

  if (loaded) {
    REQUIRE(HighsExternalDeps::isAvailable());
  } else {
    REQUIRE(!HighsExternalDeps::isAvailable());
  }
}
