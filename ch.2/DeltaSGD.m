% learnig rule
function W = DeltaSGD(W, X, D)  % W: original weight X: input D: right result
  alpha = 0.9;     % learning rate 
  
  N = 4;  
  for k = 1:N
    x = X(k, :)';
    d = D(k);

    v = W*x;                % two vectors dot product
    y = Sigmoid(v);
    
    e     = d - y;  
    delta = y*(1-y)*e;   
  
    dW = alpha*delta*x;     % delta rule: update according to error & input value
    
    W(1) = W(1) + dW(1); 
    W(2) = W(2) + dW(2);
    W(3) = W(3) + dW(3);    
  end                       % update at each iteration
end