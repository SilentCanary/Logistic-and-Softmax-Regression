#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string>
#include <Eigen/Dense>

#define EPSILON 1e-7
using namespace std;
using namespace Eigen;

void read_csv(vector<vector<double>>&data,string filename)
{
    ifstream file(filename);
    string line;
    getline(file,line);
    while(getline(file,line))
    {
        stringstream ss(line);
        string entry;
        vector<double>row;
        while (getline(ss,entry,','))
        {
            row.push_back(stod(entry));
        }
        row.insert(row.begin(),1.0);
        data.push_back(row);
    }
}

VectorXd hypothesis_function(MatrixXd& training_data,MatrixXd& thethas)
{
    MatrixXd z=training_data*thethas;
    VectorXd h=1/(1+(-z.array()).exp());
    return h;
}

double cost_function(MatrixXd & training_data,MatrixXd& thetha,VectorXd& y)
{
    int m=training_data.rows();
    VectorXd h=hypothesis_function(training_data,thetha);
    VectorXd term1=y.array()*h.array().log();
    VectorXd term2=(1-y.array())*((1-h.array()).log());
    double cost=(term1+term2).sum();
    return cost;
}

double gradient(MatrixXd& training_data,MatrixXd& thetha,VectorXd& y,int j)
{
    VectorXd h=hypothesis_function(training_data,thetha);
    VectorXd first_term=y-h;
    double grad=first_term.dot(training_data.col(j));
    return grad;
}

void gradient_ascent(MatrixXd& training_data,MatrixXd& thetha,VectorXd& y)
{
    int n=thetha.rows();
    double alpha=0.01;
    int i=0;
    double old_cost=cost_function(training_data,thetha,y);
    while(true)
    {
        for(int j=0;j<n;j++)
        {
            double grad=gradient(training_data,thetha,y,j);
            thetha(j,0)=thetha(j,0)+(alpha*grad);
        }
        double new_cost=cost_function(training_data,thetha,y);
        if(new_cost-old_cost<EPSILON)
        {
            cout<<"batch gradient ascent completed .. "<<endl;
            return;
        }
        old_cost=new_cost;
        if(i%1000==0)
        {
            for(int j=0;j<n;j++)
            {
                cout<<thetha(j,0)<<" ";
            }
            cout<<"cost : "<<new_cost<<endl;
            cout<<endl;
        }
        i++;
    }
}

VectorXd predict(MatrixXd& x,MatrixXd& thetha)
{
    VectorXd h=hypothesis_function(x,thetha);
    VectorXd predictions(h.rows());
    for (int i = 0; i < h.rows(); i++)
    {
        predictions(i) = (h(i) >= 0.5) ? 1 : 0;
    }
    return predictions;
}

double rmse(VectorXd& predicitions,VectorXd y)
{
    int m=predicitions.rows();
    int correct = 0;
    for (int i = 0; i < m; ++i)
    {
        if (predicitions(i) == y(i))
            ++correct;
    }
    return static_cast<double>(correct) /m;
}

void evaluate_metrics(const VectorXd& predictions, const VectorXd& y_true) {
    int TP = 0, TN = 0, FP = 0, FN = 0;
    int m = predictions.size();

    for (int i = 0; i < m; ++i) {
        int pred = static_cast<int>(predictions(i));
        int actual = static_cast<int>(y_true(i));

        if (pred == 1 && actual == 1) TP++;
        else if (pred == 0 && actual == 0) TN++;
        else if (pred == 1 && actual == 0) FP++;
        else if (pred == 0 && actual == 1) FN++;
    }

    double precision = TP + FP == 0 ? 0 : static_cast<double>(TP) / (TP + FP);
    double recall = TP + FN == 0 ? 0 : static_cast<double>(TP) / (TP + FN);
    double f1 = (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall);

    cout << "Confusion Matrix:\n";
    cout << "TP: " << TP << ", FP: " << FP << "\n";
    cout << "FN: " << FN << ", TN: " << TN << "\n\n";
    cout << "Precision: " << precision << endl;
    cout << "Recall: " << recall << endl;
    cout << "F1 Score: " << f1 << endl;
}


int main()
{
    vector<vector<double>>training_data;
    vector<vector<double>>test_data;
    read_csv(training_data,"train.csv");
    read_csv(test_data,"test.csv");

    int m=training_data.size();
    int n=training_data[0].size();
    MatrixXd training_data_matrix(m,n-1);
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n-1;j++)
        {
            training_data_matrix(i,j)=training_data[i][j];
        }
    }
    
    VectorXd y(m);
    for(int i=0;i<m;i++)
    {
        y(i)=training_data[i][n-1];
    }
    MatrixXd thetha=MatrixXd::Zero(n-1,1);
    gradient_ascent(training_data_matrix,thetha,y);
    int m2=test_data.size();
    int n2=test_data[0].size();
    MatrixXd test_data_matrix(m2,n2-1);

    VectorXd y_test(m2);
    for(int i=0;i<m2;i++)
    {
        for(int j=0;j<n2-1;j++)
        {
            test_data_matrix(i,j)=test_data[i][j];
        }
        y_test(i)=test_data[i][n2-1];
    }
    VectorXd predictions=predict(test_data_matrix,thetha);
    evaluate_metrics(predictions, y_test);
    return 0;
}
