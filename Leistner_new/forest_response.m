function [Yf, P] = forest_response(Yt, classes)

Yf = mode(Yt, 1)';

P = zeros(size(Yt,2), length(classes));

for i=1:length(classes),
    P(:, i) = transpose(sum(Yt==classes(i),1)/size(Yt,1));
end