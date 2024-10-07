#include <iostream>
#include <Eigen>
#include <math.h>
#include <vector>
#include <fstream>
#include <sstream>
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


void read_csv(const string& filename, vector<Vector3d>& inputs, vector<int>& labels)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cout << "COULDN'T OPEN THE FILE!!" << endl;
        return; // Exit the function if the file cannot be opened
    }

    string line;
    getline(file, line); // Skip the header line

    while (getline(file, line))
    {
        stringstream ss(line);
        Vector3d input;
        string field;
        for (int i = 0; i < 3; i++)
        {
            if (getline(ss,field, ','))
            {
                input(i) = stod(field);
            }
            else
            {
                cout << "Missing features in line " << line << endl;
            }
        }

        // Read label
        if (getline(ss,field, ','))
        {
            labels.push_back(stoi(field)); // Convert to int and store in labels vector
        }
        else
        {
            cout << "Label is missing : " << line << endl;
        }

        inputs.push_back(input); // Store the input vector
    }
}



int main()
{
    LogisticRegression lr;
    vector<Eigen::Vector3d> train_set;
    vector<int> train_labels;
    read_csv("classification_data.csv",train_set,train_labels);
    lr.gradient_descent(train_set, train_labels, 0.01, 5000);

    cout<<"Learned thetas: \n" << lr.get_thethas()<<endl;

    vector<Eigen::Vector3d>test_set;  
    vector<int>test_labels;
    read_csv("classification_test_data.csv",test_set,test_labels);

    double mse=0;
    int m=test_labels.size();
    for (int i = 0; i < m; i++)
    {
        double prediction=lr.predict(test_set[i]);
        cout << "Prediction for new input: " << prediction <<endl;
        if(prediction>0.5)
        {
            cout<<"Prediction result: Yes"<<endl;
        }
        else
        {
            cout<<"Prediction result: No"<<endl;
        }
        double err=prediction-test_labels[i];
        mse+=pow(err,2);
    }
    mse=mse/m;
    cout<<"Mean Sqaured Error (MSE) on model is : "<<mse<<endl;

    return 0;
}
