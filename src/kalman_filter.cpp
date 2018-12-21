#include <iostream>

#include "kalman_filter.h"

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

#define PI 3.14159265358979323846

KalmanFilter::KalmanFilter() = default;

KalmanFilter::~KalmanFilter() = default;

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_*x_;
  P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  MatrixXd I(4,4);

  I << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  MatrixXd H_t = H_.transpose();

  VectorXd z_pred = H_*x_;

  VectorXd y = z - z_pred;

  MatrixXd S = H_*P_*H_t + R_;
  MatrixXd K = P_*H_t*S.inverse();

  x_ = x_ + K*y;
  P_ = (I - K*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  double distance = sqrt(px*px + py*py);

  if(fabs(distance) < 1e-6) {
    cout << "KalmanFilter::UpdateEKF --- divide by zero!" << endl;
    exit(-1);
  }

  VectorXd z_pred(3);

  MatrixXd I(4,4);

  I << 1, 0, 0, 0,
       0, 1, 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;

  MatrixXd H_t = H_.transpose();

  z_pred << distance, atan2(py,px), (px*vx + py*vy)/distance;

  VectorXd y = z - z_pred;

  double phi = y(1);

  while(phi<-PI) {
    phi += 2*PI;
  }

  while(phi>PI) {
    phi -= 2*PI;
  }

  y(1) = phi;

  MatrixXd S = H_*P_*H_t + R_;
  MatrixXd K = P_*H_t*S.inverse();

  x_ += K*y;
  P_ = (I - K*H_)*P_;

}
