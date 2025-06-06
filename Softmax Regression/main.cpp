#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>

using namespace std;
using namespace Eigen;

void read_csv(vector<vector<double>>&data,string filename)
{
    ifstream file(filename);
    string line;
    while(getline(file,line))
    {
        stringstream ss(line);
        string entry;
        vector<double>row;
        bool skip=false;
        while(getline(ss,entry,','))
        {
            if(entry.empty())
            {
                skip=true;
                break;
            }
            row.push_back(stod(entry));
        }
        if(!skip && !row.empty())
        {
            row.insert(row.begin()+3, 1.0);
            data.push_back(row);
        }
    }
}

MatrixXd softmax(MatrixXd& z)
{
    MatrixXd exp_z=z.array().exp();
    VectorXd row_sums=exp_z.rowwise().sum();
    MatrixXd p_hat=exp_z.array().colwise()/row_sums.array();
    return p_hat;
}

double compute_cost(MatrixXd& y,MatrixXd& p_hat)
{
    int m=y.rows();
    double cost=0;
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<y.cols();j++)
        {
            if(y(i,j)==1)
            {
                cost+=log(p_hat(i,j)+1e-15);
            }
        }
    }
    return -cost/m;
}

MatrixXd compute_gradients(MatrixXd&x ,MatrixXd&y,MatrixXd& p_hat)
{
    int m=x.rows();
    MatrixXd grad=(x.transpose()*(p_hat-y))/m;
    return grad;
}

void gradient_descent(MatrixXd& thethas,MatrixXd& x,MatrixXd& y)
{
    double alpha=0.01;
    MatrixXd z=x*thethas;
    MatrixXd p_hat=softmax(z);
    double old_cost=compute_cost(y,p_hat);
    int i=0;
    while (true)
    {
        MatrixXd grad=compute_gradients(x,y,p_hat);
        thethas=thethas-(alpha*grad);
        z=x*thethas;
        p_hat=softmax(z);
        double new_cost=compute_cost(y,p_hat);
        if(abs(new_cost-old_cost)<1e-7)
        {
            cout<<"gradient descent completed";
            return;
        }
        old_cost=new_cost;
        if(i%1000==0)
        {
            cout<<"iteration : "<<i<<" cost : "<<new_cost<<endl;
        }
        i++;
    }
    
}

VectorXi predict(const MatrixXd& x, const MatrixXd& thethas) {
    MatrixXd z = x * thethas;
    MatrixXd p_hat = softmax(z);

    VectorXi preds(p_hat.rows());
    for (int i = 0; i < p_hat.rows(); i++) {
        p_hat.row(i).maxCoeff(&preds(i));
    }
    return preds;
}


double compute_accuracy(const VectorXi& preds, const MatrixXd& y_true) {
    int correct = 0;
    for (int i = 0; i < preds.size(); i++) {
        int actual_label = -1;
        y_true.row(i).maxCoeff(&actual_label);  
        if (preds(i) == actual_label) {
            correct++;
        }
    }
    return static_cast<double>(correct) / preds.size();
}


//did gpt for these two functions since i get confused writing matrix and f1 score
void compute_confusion_matrix(const VectorXi& preds, const MatrixXd& y_true, MatrixXi& conf_matrix) {
    conf_matrix = MatrixXi::Zero(3, 3);  // Assuming 3 classes

    for (int i = 0; i < preds.size(); ++i) {
        int actual = -1;
        y_true.row(i).maxCoeff(&actual);
        int predicted = preds(i);
        conf_matrix(actual, predicted)++;
    }
}

void compute_f1_score(const MatrixXi& conf_matrix) {
    double precision_sum = 0.0, recall_sum = 0.0, f1_sum = 0.0;

    for (int i = 0; i < 3; ++i) {
        double TP = conf_matrix(i, i);
        double FP = conf_matrix.col(i).sum() - TP;
        double FN = conf_matrix.row(i).sum() - TP;

        double precision = (TP + 1e-15) / (TP + FP + 1e-15);
        double recall = (TP + 1e-15) / (TP + FN + 1e-15);
        double f1 = 2 * precision * recall / (precision + recall + 1e-15);

        cout << "Class " << i << " -- Precision: " << precision << " Recall: " << recall << " F1 Score: " << f1 << endl;

        precision_sum += precision;
        recall_sum += recall;
        f1_sum += f1;
    }

    cout << "Macro-averaged F1 Score: " << f1_sum / 3 << endl;
}

int main()
{
    vector<vector<double>>training_data;
    vector<vector<double>>test_data;
    read_csv(training_data,"train2.csv");
    read_csv(test_data,"test2.csv");
    int m=training_data.size();
    int m2=test_data.size();
    int n=training_data[0].size();
    int n2=test_data[0].size();
    MatrixXd x(m,n-3); //first 3 are actually y
    MatrixXd x_test(m2,n2-3);
    for(int i=0;i<m;i++)
    {
        for(int j=3;j<n;j++)
        {
            x(i,j-3)=training_data[i][j];
        }
    }
    for(int i=0;i<m2;i++)
    {
        for(int j=3;j<n2;j++)
        {
            x_test(i,j-3)=test_data[i][j];
        }
    }
    MatrixXd y(m,3);
    MatrixXd y_test(m2,3);
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<3;j++)
        {
            y(i,j)=training_data[i][j];
        }
    }
    for(int i=0;i<m2;i++)
    {
        for(int j=0;j<3;j++)
        {
            y_test(i,j)=test_data[i][j];
        }
    }
    MatrixXd thethas=MatrixXd::Zero(n-3,3);
    gradient_descent(thethas,x,y);
    VectorXi predictions=predict(x_test,thethas);
    cout<<"accuracy is : "<<compute_accuracy(predictions,y_test);
    MatrixXi confusion_matrix;
    compute_confusion_matrix(predictions, y_test, confusion_matrix);
    cout << "\nConfusion Matrix:\n" << confusion_matrix << endl;
    
    compute_f1_score(confusion_matrix);

    return 0;
}