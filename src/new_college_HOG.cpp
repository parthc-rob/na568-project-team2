#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>
#include <opencv2/objdetect.hpp>

#include <DBoW2/DBoW2.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "frame_descriptor_SURF.h"
#include "utils.h"

using namespace std;
float score(const vector<float>& des1, const vector<float>& des2);
// float score(const& des1, const& des2);
int main(int argc, char* argv[]) {
  // std::string vocabulary_path("data/ORBvoc.txt");
  // slc::FrameDescriptor descriptor(vocabulary_path);
  
  std::string dataset_folder("data");
  const auto filenames = load_filenames(dataset_folder);

  if (filenames.size() == 0) {
    std::cerr << "No images found in " << dataset_folder << "\n";
    exit(1);
  }

  std::cerr << "Processing " << filenames.size() << " images\n";

  // Will hold BoW representations for each frame
  // std::vector<DBoW2::BowVector> bow_vecs;
  std::vector<std::vector<float>>descriptors;
  // bow_vecs.reserve(filenames.size());
  // cv::HOGDescriptor hog(cv::Size _winSize(16,32), );
  cv::HOGDescriptor* hog=new cv::HOGDescriptor(cvSize(16,32),cvSize(16,16),cvSize(16,16),cvSize(8,8),9,1);
  // hog.winSize = cv::imread().size();

  for (unsigned int img_i = 0; img_i < filenames.size(); img_i++) {
    auto img_filename = dataset_folder + "/Images/" + filenames[img_i];
    auto img = cv::imread(img_filename,0);
    if (img.empty()) {
      std::cerr << std::endl << "Failed to load: " << img_filename << std::endl;
      exit(1);
    }
    // hog.winSize = img.size();

    std::vector<cv::Point>positions;
    // cv::Size s1(16,16);
    // cv::Size s2(4,4);
    positions.push_back(cv::Point(img.cols / 2, img.rows / 2));
    std::vector<float> descriptor;
    hog->compute(img,descriptor,cv::Size(),cv::Size(),positions);
    // std::cerr << img_filename << "\n";
    std::cout<< "The dimension of HOG descriptor is: " << descriptor.size() << std::endl;

    // Get a HOG descriptor of the current image
    // DBoW2::BowVector bow_vec;
    // descriptor.describe_frame(img, bow_vec);
    // bow_vecs.push_back(bow_vec);
    descriptors.push_back(descriptor);
  }

  std::cerr << "\nWriting output...\n";

  std::string output_path("out/confusion_matrix.txt");
  std::ofstream of;
  of.open(output_path);
  if (of.fail()) {
    std::cerr << "Failed to open output file " << output_path << std::endl;
    exit(1);
  }
  
  // Compute confusion matrix
  // i.e. the (i, j) element of the matrix contains the distance
  // between the BoW representation of frames i and j
  std::cout << "The number of descriptors is " <<descriptors.size() <<std::endl;
  cv::Mat score_mat; 
  score_mat.create(descriptors.size(),descriptors.size(),5);

  for (int i = 0; i < descriptors.size(); i++) {
    for (int j = 0; j < descriptors.size(); j++) {
      // score_mat.at<float>(i,j) = 1/(1+score(descriptors[i],descriptors[j]));
      score_mat.at<float>(i,j) = -score(descriptors[i],descriptors[j]);
      std::cout<< "The score is: " << score_mat.at<float>(i,j) << std::endl;
    }
  }
  // Normalize it now
  cv::Mat score_mat_1;
  cv::normalize(score_mat, score_mat_1, 1.0, 0.0, cv::NORM_MINMAX);

 for (int i = 0; i < descriptors.size(); i++) {
    for (int j = 0; j < descriptors.size(); j++) { 
      // of << descriptor.vocab_->score(bow_vecs[i], bow_vecs[j]) << " ";
      of << score_mat_1.at<float>(i,j) << " ";
    }
    of << "\n";
  }
  // of.close();
  std::cerr << "Output done\n";
}


float score(const vector<float>& des1, const vector<float>& des2){
  float sum = 0;
  float dist = 0;
  for (int i=0; i < des1.size(); i++){
      sum += pow((des1[i] - des2[i]),2);
    }
  dist = pow(sum,0.5);

  return dist;
}