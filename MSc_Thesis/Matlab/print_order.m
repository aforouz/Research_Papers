function print_order(config_data)

order_config = config_data.order_config;
save(config_data.print_config.address + "error", "config_data");

file_ID = fopen(config_data.print_config.address + "order.txt", "w");

fprintf(file_ID, "============================================================\n\n"); 

order_t_x_y = [0; log(order_config.error_step(1:end-1)./order_config.error_step(2:end))/log(order_config.Bxy)];
fprintf(file_ID, "M\t\tN\t\tError\t\tRate\tTime(s)\n");
for dxyt = 1:length(order_config.Dxy)
    fprintf(file_ID, "%-6d\t%-6d\t%0.3E\t%0.2f\t%0.5f\n", ...
        power(order_config.Bxy,order_config.Dxy(dxyt)), power(order_config.Bt,order_config.Dt(dxyt)), ...
        order_config.error_step(dxyt), order_t_x_y(dxyt), order_config.time_step(dxyt));
end

fprintf(file_ID, "\n============================================================\n\n"); 

fclose(file_ID);


%{
_x_y_" + order_config.Bxy + "_" + order_config.Dxy(1) + "_" + order_config.Dxy(end) + ...
                    "_t_"  + order_config.Bt + "_" + order_config.Dt(1) + "_" + order_config.Dt(end) + "_" + order_config.print_config.solver_method + "

%}