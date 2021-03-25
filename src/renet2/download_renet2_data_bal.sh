#!/bin/bash
set -x

# download data
## download abstract dataset [dir: renet2/data/abs_data]
(cd data/abs_data && cp /autofs/bal31/jhsu/resources/RENET2/renet2_abs_data.tar.gz . && tar -xf renet2_abs_data.tar.gz && rm renet2_abs_data.tar.gz )

## download full-text dataset [dir: renet2/data/ft_data]
(cd data/ft_data && cp /autofs/bal31/jhsu/resources/RENET2/renet2_ft_data.tar.gz . && tar -xf renet2_ft_data.tar.gz && rm renet2_ft_data.tar.gz )
(cd data/ft_info && cp /autofs/bal31/jhsu/resources/RENET2/renet2_ft_data_info.tar.gz . && tar -xf renet2_ft_data_info.tar.gz && rm renet2_ft_data_info.tar.gz )

## download gene-disease assoications found from PMC and LitCovid [dir: renet2/data/pmc_litcovid]
(cd data && cp /autofs/bal31/jhsu/resources/RENET2/renet2_gda_pmc_litcovid_rst.tar.gz . && tar -xf renet2_gda_pmc_litcovid_rst.tar.gz && rm renet2_gda_pmc_litcovid_rst.tar.gz)

# download trained model
## download trained abstract model [dir: renet2/models/abs_models]
(cd models && cp /autofs/bal31/jhsu/resources/RENET2/renet2_trained_models_abs.tar.gz . && tar -xf renet2_trained_models_abs.tar.gz && rm renet2_trained_models_abs.tar.gz)

## download trained abstract model [dir: renet2/models/ft_models]
(cd models && cp /autofs/bal31/jhsu/resources/RENET2/renet2_trained_models_ft.tar.gz . && tar -xf renet2_trained_models_ft.tar.gz && rm renet2_trained_models_ft.tar.gz)

