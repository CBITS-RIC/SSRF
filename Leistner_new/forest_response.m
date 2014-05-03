function [Yf, Pf] = forest_response(Yt, Pt, classes)

Yf = mode(Yt, 1)';

Pf = squeeze(mean(Pt,1));