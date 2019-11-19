function [features] = awet_features_extract(I, I_annotations)
    
    global awet;

    I = uint8(imresize(I, [awet.current_parameter_set.s, awet.current_parameter_set.s]));
    
    a = [1 -1];
    b = [1; -1];
    c = [1 0; 0 -1];
       
    aa = conv2(I, a, 'same');
    bb = conv2(I, b, 'same');
    cc = conv2(I, c, 'same');
    
    features = uint8([reshape(aa, 1, []) reshape(bb, 1, []) reshape(cc, 1, [])]);
    
 end
