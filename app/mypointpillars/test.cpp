#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <cmath>
#include <chrono>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/configurable_dpu_task.hpp>
#include <vitis/ai/pointpillars_nuscenes.hpp>
#include <fstream>
#include <map>
#include <vector>
#include <math.h>
#include <sys/stat.h>
#include "voxelize.h"

using namespace std;


vector<float> loadPoints(string points_file_name){
    std::vector<float> points;
    struct stat file_stat;
    if (stat(points_file_name.c_str(), &file_stat) != 0) {
        std::cerr << "file:" << points_file_name << " state error!" << std::endl;
        exit(-1);
    }
    auto file_size = file_stat.st_size;
    points.resize(file_size / 4);
    CHECK(std::ifstream(points_file_name).read(reinterpret_cast<char *>(points.data()), file_size).good());
    return points;
}

class PointPillarsRunner{
    public:
        std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_0;
        std::unique_ptr<vitis::ai::ConfigurableDpuTask> model_1;
        std::unique_ptr<vitis::ai::PointPillarsNuscenesPostProcess> postprocessor;
        uint32_t points_dim;
        uint32_t model_in_channels;
        std::vector<float> input_mean;
        std::vector<float> input_scale;
        int max_points_num;
        int max_voxels_num;

    public: PointPillarsRunner(std::string model_name_0, std::string model_name_1){
        //model
        bool need_preprocess = false;
        auto attrs = xir::Attrs::create();
        this->model_0 = vitis::ai::ConfigurableDpuTask::create(model_name_0, attrs.get(), need_preprocess);
        this->model_1 = vitis::ai::ConfigurableDpuTask::create(model_name_1, attrs.get(), need_preprocess);
        this->postprocessor = (vitis::ai::PointPillarsNuscenesPostProcess::create(
            this->model_1->getInputTensor()[0],
            this->model_1->getOutputTensor()[0],
            this->model_0->getConfig()));
        this->points_dim = this->model_0->getInputTensor()[0][0].channel;

        // std::vector<float> points_range;
        // std::copy(model_0->getConfig().pointpillars_nus_param().point_cloud_range().begin(),
        //         model_0->getConfig().pointpillars_nus_param().point_cloud_range().end(),
        //         std::back_inserter(points_range));
        this->model_in_channels = model_0->getConfig().pointpillars_nus_param().in_channels();

        std::copy(this->model_0->getConfig().kernel(0).mean().begin(), this->model_0->getConfig().kernel(0).mean().end(), std::back_inserter(this->input_mean));
        std::copy(this->model_0->getConfig().kernel(0).scale().begin(), this->model_0->getConfig().kernel(0).scale().end(), std::back_inserter(this->input_scale));
        this->max_points_num = this->model_0->getConfig().pointpillars_nus_param().max_points_num();
        this->max_voxels_num = this->model_0->getConfig().pointpillars_nus_param().max_voxels_num();

        auto input_tensor_scale = vitis::ai::library::tensor_scale(this->model_0->getInputTensor()[0][0]);
        for (auto i = 0u; i < this->input_scale.size(); ++i) {
            this->input_scale[i] *= input_tensor_scale;
        }
    }

    private: void scatter(const std::vector<int> &coors, int coors_dim, const int8_t *input_data, float input_scale,
             int8_t *output_data, float output_scale, uint32_t in_channels, int nx, int ny) {
        //auto size = w * h * c;
        //auto coors_shape = coors.shape; // [40000, 4] or [num, 4]
        auto coors_num = coors.size() / coors_dim;

        bool copy_directly = (std::abs(input_scale * output_scale -1) < 0.0001);

        for (auto i = 0u; i < coors_num; ++i) {
            auto index = coors[i * coors_dim + 2] * nx + coors[i * coors_dim + 3];
            auto ibegin = input_data + i * in_channels;
            auto iend = ibegin + in_channels;
            auto obegin = output_data + index * in_channels;
            if (copy_directly) {
                std::memcpy(obegin, ibegin, in_channels);
            } else {
                std::transform(ibegin, iend, obegin, [=](int8_t in)->int8_t {return (int)(in * input_scale * output_scale);});
            }
        }
    }

    public: vitis::ai::PointPillarsNuscenesResult Run(vector<float>& points){
        size_t batch = 1;
        int num = 1;
        auto batch_idx = 0u;

        std::vector<int> coors;
        auto input_tensor_dim = this->points_dim;
        auto model_0_input_size = this->model_0->getInputTensor()[0][0].size / batch;

        std::memset(this->model_0->getInputTensor()[0][0].get_data(batch_idx), 0, model_0_input_size);
        auto input_ptr = (int8_t *)this->model_0->getInputTensor()[0][0].get_data(batch_idx);
        // coors = voxelizer_->voxelize(points, input_tensor_dim, input_ptr, model_0_input_size);
        coors = voxelize(this->input_mean, this->input_scale, this->max_points_num, this->max_voxels_num, points, input_tensor_dim, input_ptr, model_0_input_size);
        this->model_0->run(0);

        auto model_1_input_tensor_size = this->model_1->getInputTensor()[0][0].size;
        auto model_1_input_size = model_1_input_tensor_size / batch;
        std::memset(this->model_1->getInputTensor()[0][0].get_data(batch_idx), 0, model_1_input_size);
        auto coors_dim = 4;
        auto nx = this->model_1->getInputTensor()[0][0].width;
        auto ny = this->model_1->getInputTensor()[0][0].height;
        this->scatter(coors, coors_dim,
            (int8_t *)this->model_0->getOutputTensor()[0][0].get_data(batch_idx),
            vitis::ai::library::tensor_scale(this->model_0->getOutputTensor()[0][0]),
            (int8_t *)this->model_1->getInputTensor()[0][0].get_data(batch_idx),
            vitis::ai::library::tensor_scale(this->model_1->getInputTensor()[0][0]),
            this->model_in_channels, nx, ny);

        this->model_1->run(0);
        auto results = this->postprocessor->postprocess(num);
        return results[batch_idx];
    }
};


int main(int argc, char *argv[]){
    if (argc != 3) {
        cerr << "usage: ./test <.bin> <logfile>" << endl;
    }
    #ifdef RISCV
    riscv_init();
    #endif
    string pcdfile = argv[1];
    string logfile = argv[2];
    vector<float> points = loadPoints(pcdfile);
    auto runner = PointPillarsRunner("net0.xmodel", "net1.xmodel");
    std::chrono::system_clock::time_point  t1, t2;
    t1 = std::chrono::system_clock::now();
    auto ret = runner.Run(points);
    t2 = std::chrono::system_clock::now();
    double elapsed1 = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

    cout << "predict: " << elapsed1 << "[ms]" << endl;
    ofstream logstream(logfile);
    for (auto c = 0u; c < 10; ++c) {
      for (auto i = 0u; i < ret.bboxes.size(); ++i) {
        if (ret.bboxes[i].label != c) {
          continue;
        }
        logstream << "label: " << ret.bboxes[i].label;
        logstream << " bbox: ";
        for (auto j = 0u; j < ret.bboxes[i].bbox.size(); ++j) {
          logstream << ret.bboxes[i].bbox[j] << " ";
        }
        logstream << "score: " << ret.bboxes[i].score;
        logstream << endl;
      }
    }
    logstream.close();
    cout << "predicted log is written in " << logfile << endl;
}