#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <string>

#include <lkh_mtsp_solver/lkh3_interface.h>
#include <lkh_mtsp_solver/SolveMTSP.h>

using std::string;

std::string mtsp_root_dir_;

bool mtspCallback(
    lkh_mtsp_solver::SolveMTSP::Request& req, lkh_mtsp_solver::SolveMTSP::Response& res)
{
  // prob=1 keeps the legacy agent-0 files compatible; higher values select
  // an independent per-agent file without changing the service MD5.
  const int agent_id = static_cast<int>(req.prob) - 1;
  if (agent_id >= 0 && agent_id < 2) {
    const std::string stem =
        agent_id == 0 ? "atsp_tour" : "atsp_tour_agent_" + std::to_string(agent_id);
    const std::string par_file = mtsp_root_dir_ + "/" + stem + ".par";
    solveMTSPWithLKH3(par_file.c_str());
    return true;
  }
  ROS_ERROR("Unsupported ATSP problem code: %u", req.prob);
  return false;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tsp_node");
  ros::NodeHandle nh("~");

  // Read mtsp file dir
  std::string tsp_dir;
  nh.param("exploration/tsp_dir", tsp_dir, std::string("null"));

  mtsp_root_dir_ = tsp_dir;

  string service_name = "/solve_tsp";
  ros::ServiceServer mtsp_server = nh.advertiseService(service_name, mtspCallback);

  ROS_WARN("TSP server is ready.");
  ros::spin();

  return 1;
}
