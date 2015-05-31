% project onto the non-convex constraint of the polar approximation
% Forward-backward splitting
function u_new = ProjPolar(u, r, theta)   
% lambda is the l1 regularization coeff

x = u(:,1);
y = u(:,2);
z = u(:,3);

for i = 1 : size(u,1)
    % we first determine what theta_proj should be using y and z
    % coordinates
%     if (x(i) + r*y(i)) <= 0            % if x+y <= 0  then we project to the origin
%         x(i) = 0;
%         y(i) = 0;
%         z(i) = 0;
%     else
        if (abs(atan(z(i)/y(i))) <= theta)  % if we are within the sector
            theta_proj = atan(z(i)/y(i));
        elseif (z(i) > 0)                   % otherwise the projection will be at one of the endpoints of the arc
            theta_proj = theta;
        elseif (z(i) < 0)
            theta_proj = -theta;
        elseif (z(i) == 0)                  % when z(i) = 0, we pick either endpoint at random
            if rand <= 0.5
                theta_proj = theta;
            else
                theta_proj = -theta;
            end
        end
        % now we project onto the line specified by [1/r cos(theta_proj)
        % sin(theta_proj)]
        unit_vec = [1/r cos(theta_proj) sin(theta_proj)]'/norm([1/r cos(theta_proj) sin(theta_proj)]');
        projection_length = dot([x(i) y(i) z(i)]', unit_vec);
        x(i) = projection_length * unit_vec(1);
        y(i) = projection_length * unit_vec(2);
        z(i) = projection_length * unit_vec(3);
        
        if x(i) <= 0
            x(i) = 0;
            y(i) = 0;
            z(i) = 0;
        end
        
%     end

    
end

u_new(:,1) = x;
u_new(:,2) = y;
u_new(:,3) = z;
