#pragma once

#include <DBoW2/DBoW2.h>

#include <opencv/cv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

namespace slc {

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;

struct FrameDescriptor {
  /*
   * Computes a global representation for an image by using
   * the SURF feature descriptor in OpenCV and the bag of
   * words approach.
   */
  FrameDescriptor(const std::string& vocabulary_path) {
    std::cout << "Loading vocabulary from " << vocabulary_path << std::endl;
    // vocab_.=new ORBVocabulary();
    std::cout << "Loaded vocabulary with " << vocab_->size() << " visual words."
              << std::endl;
    vocab_->ORBVocabulary::loadFromTextFile(vocabulary_path);
  }

  void extract_orb(const cv::Mat& img, std::vector<cv::KeyPoint>& keys,
                    cv::Mat& descriptors, std::vector<std::vector<cv::Mat>>& features) {
    /* Extracts SURF interest points and their descriptors. */
    cv::Mat mask;
    static cv::Ptr<cv::ORB> orb_detector = cv::ORB::create(400);
        // cv::xfeatures2d::SURF::create(400);

    // orb_detector->setExtended(false);

    // std::vector<float> plain;
    orb_detector->detectAndCompute(img, mask, keys, descriptors);
    features.push_back(std::vector<cv::Mat>());
    changeStructure(descriptors, features.back());
    // std::cout << "Plain size is: " << plain.size() << std::endl;
    // const int L = orb_detector->descriptorSize();
    
    // descriptors.resize(plain.size() / L);
    // std::cout << "Descriptor size is: " << L << std::endl;

    // unsigned int j = 0;
    // for (unsigned int i = 0; i < plain.size(); i += L, ++j) {
    //   descriptors[j].resize(L);
    //   std::copy(plain.begin() + i, plain.begin() + i + L,
    //             descriptors[j].begin());
    // }
  };
  void changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out)
  {
    out.resize(plain.rows);
 
  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);// 每一个256bit的描述子单独占用一个cv::Mat结构
  }
  };

  void describe_frame(const cv::Mat& img, DBoW2::BowVector& bow_vec) {
    /* Transforms the feature descriptors to a BoW representation
     *  of the whole image. */

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<std::vector<cv::Mat>> features;
    extract_orb(img, keypoints, descriptors, features);
    vocab_->transform(descriptors, bow_vec);
  }

  std::unique_ptr<ORBVocabulary> vocab_;
  // ORBVocabulary* vocab_;
};
}  // namespace slc
