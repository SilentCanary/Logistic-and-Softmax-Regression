#include <iostream>
#include <Eigen>
#include <math.h>
#include <vector>
using namespace std;
using namespace Eigen ;
class LogisticRegression
{
    Vector3d thethas;
    Vector3d input_features;

    public:
    LogisticRegression()
    {
        thethas=Vector3d::Zero();
    }
    double sigmoid_function(double z)
    {
        return 1/(1+exp(-z));
    }
    double predict(const Vector3d& input_features)
    {
        double z=thethas.dot(input_features);
        return sigmoid_function(z);
    }
    double cost_function(const vector<Vector3d>& input_features,const vector<int>&y)
    {
        int m=input_features.size(); //no of training examples
        double j_thetha=0.0;
        for (int i = 0; i < m; i++)
        {
            double h_thetha=predict(input_features[i]);
            j_thetha+=y[i]*log(h_thetha)+(1-y[i])*log(1-h_thetha);
        }
        j_thetha=-j_thetha/m;
        return j_thetha;
    }
    void gradient_descent(const vector<Vector3d>&input_features,const vector<int>&y,double alpha,int iterations)
    {
        int m=input_features.size();
        for (int i = 0; i < iterations; i++)
        {
            Vector3d gradients=Vector3d::Zero();
            for (int j = 0; j < m; j++)
            {
                double h_thetha=predict(input_features[j]);
                double error=h_thetha-y[j];
                gradients+=error*input_features[j];
            }
            thethas=thethas-alpha*gradients/m;
            if(i%100==0)
            {
               cout<<"Cost in "<<i<<" iteration : "<<cost_function(input_features,y)<<endl;
            }
        }
        
    }
    Vector3d get_thethas()
    {
        return thethas;
    }
};

int main()
{
    LogisticRegression lr;
    vector<Eigen::Vector3d> inputs = {
        Eigen::Vector3d(1.0, 2.0, 3.0),  
        Eigen::Vector3d(1.0, 4.0, 5.0), 
        Eigen::Vector3d(1.0, 7.0, 8.0)   
    };
    vector<int> labels = {0, 1, 1};

    lr.gradient_descent(inputs, labels, 0.01, 5000);

    cout<<"Learned thetas: \n" << lr.get_thethas()<<endl;
    Eigen::Vector3d new_input(1.0, 7.0, 8.0);  
    double prediction = lr.predict(new_input);
    cout << "Prediction for new input: " << prediction <<endl;

    return 0;
}
