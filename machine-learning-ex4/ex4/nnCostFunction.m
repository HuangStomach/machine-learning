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
z1 = sigmoid(Theta1 * X');
a2 = [ones(1, size(z1, 2)); z1];
a3 = sigmoid(Theta2 * a2);
h = a3;
% 得到输出层并且得到消费函数

Y = zeros(num_labels, m);
% 将Y矩阵化
% 每列都是0，当前的y值的行变为1
for i = 1: num_labels,
    Y(i, :) = (y == i);
end;

% 将 Theta1 和  Theta2 的 theta0 去掉或者变为0
Theta1Reg = Theta1(:,2:size(Theta1,2));
Theta2Reg = Theta2(:,2:size(Theta2,2));

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

for k = 1: m
    a1 = X(k,:);
    z2 = Theta1 * a1';
    a2 = sigmoid(z2);
    a2 = [1 ; a2];
    a3 = sigmoid(Theta2 * a2);

    % 使用反推算法计算误差
    d3 = a3 - Y(:,k);
    
    % 每一个 d 都为 后一个 d 与 theta并且通过对z2求导得来
    z2 = [1; z2];
    d2 = (Theta2' * d3) .* sigmoidGradient(z2);
    % 去掉 d20
    d2 = d2(2: end);

    Theta2_grad = (Theta2_grad + d3 * a2');
    Theta1_grad = (Theta1_grad + d2 * a1); % input层不用求导
end;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% -------------------------------------------------------------

% 0 列进行不参与 lamda 的规范化
Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;

Theta1_grad(2, end) = Theta1_grad(2, end) ./ m + lambda / m * Theta1_grad(2, end);
Theta2_grad(2, end) = Theta2_grad(2, end) ./ m + lambda / m * Theta1_grad(2, end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
