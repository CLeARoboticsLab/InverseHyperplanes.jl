"2D satellite dynamics"
function cwh_satellite_2D(; dt = 5.0, n = 0.001, m = 100.0, kwargs...)
    # Layout is x := (px, py, vx, vy) and u := (ax, ay).
    time_invariant_linear_dynamics(;
    A = [ 4-3*cos(n*dt)         0  1/n*sin(n*dt)     2/n*(1-cos(n*dt))        ;
             6*(sin(n*dt)-n*dt) 1 -2/n*(1-cos(n*dt)) 1/n*(4*sin(n*dt)-3*n*dt) ;   
             3*n*sin(n*dt)      0  cos(n*dt)         2*sin(n*dt)              ;
            -6*n*(1-cos(n*dt))  0 -2*sin(n*dt)       4*cos(n*dt)-3            ],

    B = 1/m*[ 1/n^2(1-cos(n*dt))     2/n^2*(n*dt-sin(n*dt))       ;
             -2/n^2*(n*dt-sin(n*dt)) 4/n^2*(1-cos(n*dt))-3/2*dt^2 ;
              1/n*sin(n*dt)          2/n*(1-cos(n*dt))            ;
             -2/n*(1-cos(n*dt))      4/n*sin(n*dt)-3*dt           ;],
        kwargs...,
    )
end