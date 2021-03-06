function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% 首选计算出每一个隐藏层
X = [ones(m, 1), X];
a1 = X;
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;
% 得到输出层并且得到消费函数

% 将Y矩阵化
% 每列都是0，当前的y值的行变为1
tmp = eye(num_labels);
Y = tmp(y, :);

% 将 Theta1 和  Theta2 的 theta0 去掉或者变为0
Theta1Reg = Theta1(:, 2: end);
Theta2Reg = Theta2(:, 2: end);

J = sum(sum(-Y .* log(h) - (1 - Y) .* log(1 - h))) / m;
J = J + lambda / (2 * m) * (sum(sum(Theta1Reg .^ 2)) + sum(sum(Theta2Reg .^ 2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

for i = 1 : m
    a1_i = X(i, :);
    z2_i = a1_i * Theta1';
    a2_i = sigmoid(z2_i);
    % add bias to vector a2_i
    a2_i = [1, a2_i];
    % ======= LAYER 3 =======
    % sigmoid input (can be vector or matrix - depending on Theta2)
    z3_i = a2_i * Theta2';
    % activation unit (this can be matrix or vector, depending on output of sigmoid)
    a3_i = sigmoid(z3_i);
    % compute output error
    delta_3 = (a3_i - Y(i, :));
    % compute hidden layer error
    % ignore the bias term and calculate error for layer 2
    z2_i = [1, z2_i];
    delta_2 = (Theta2' * delta_3')' .* sigmoidGradient(z2_i);
    delta_2 = delta_2(2: end);

    % Transition matrix error accumulators
    Theta1_grad = Theta1_grad + delta_2' * a1_i;
    Theta2_grad = Theta2_grad + delta_3' * a2_i;
end;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% -------------------------------------------------------------

% 0 列进行不参与 lamda 的规范化
Theta1_grad = Theta1_grad / m + (lambda / m) * [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];
Theta2_grad = Theta2_grad / m + (lambda / m) * [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
