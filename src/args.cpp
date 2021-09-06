#include "args.hpp"



using namespace sph::driver;

Args::Args(size_t defaultIterations, const std::string &defaultOutput)
    : parser("PBF sph benchmark", ""),                                                                            //
      help(parser, "help", "Display this help menu",                                                              //
           {'h', "help"}),                                                                                        //
      completion(parser, {"complete"}),                                                                           //
      impl(parser, "impl", "Which implementation to use.",                                                        //
           {"i", "impl"}, IMPLS, IMPLS.at(DEFAULT_IMPL)),                                                         //
      list(parser, "list",                                                                                        //
           "List devices available for [impl] and exit",                                                          //
           {'l', "list"}),                                                                                        //
      verbose(parser, "verbose",                                                                                  //
              "Show details such as device tree for [impl]",                                                      //
              {'v', "verbose"}),                                                                                  //
      devices(parser, "dev",                                                                                      //
              "Allowed device list (first match only).\n"                                                         //
              "Entries could be a 0-based index or a substring of the device name ",                              //
              {'d', "devices"},                                                                                   //
              {DEFAULT_DEVICE}),                                                                                  //
      iterations(parser, "iter",                                                                                  //
                 "How many iterations to run the simulation for, use 0 for infinity.",                            //
                 {'n', "iter"},                                                                                   //
                 defaultIterations),                                                                              //
      warmup(parser, "warmup",                                                                                    //
             "How many iterations to skip for warmup before timing starts.\n"                                     //
             "This has no effect if an active visualisation is running.",                                         //
             {'w', "warmup"},                                                                                     //
             DEFAULT_WARMUP),                                                                                     //
      fp64(parser, "fp64",                                                                                        //
           "Use FP64 (double) instead of FP32 (float).\n"                                                         //
           "Not all implementations support fp64.",                                                               //
           {"fp64"}),                                                                                             //
      output(                                                                                                     //
          parser, "out",                                                                                          //
          "Directory to write the final state (cloud.ply, mesh.obj) to.\n"                                        //
          "The directory will be created if it doesn't exist, ignore or set to empty string to disable output.\n" //
          "The following templates for the filename are available: {iter}, {impl}, {type}",                       //
          {'o', "output"}, defaultOutput)                                                                         //
{
  impl.HelpDefault(DEFAULT_IMPL);
  parser.Prog("benchmark");
  parser.helpParams.width = 120;
  parser.helpParams.addChoices = true;
  parser.helpParams.addDefault = true;
}

bool Args::parse(int argc, char *argv[]) {
  try {
    parser.ParseCLI(argc, argv);
    return true;
  } catch (const args::Completion &e) {
    std::cout << e.what();
  } catch (const args::Help &) {
    std::cout << parser;
  } catch (const args::ParseError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
  }
  return false;
}

std::string Args::implName() { return IMPLS_R.at(impl.Get()); }

std::string Args::renderedOutputName() {
  std::string name = output.Get();
  name = sph::utils::replace(name, "{iter}", std::to_string(iterations.Get()));
  name = sph::utils::replace(name, "{type}", fp64 ? "fp64" : "fp32");
  name = sph::utils::replace(name, "{impl}", implName());
  return name;
}
