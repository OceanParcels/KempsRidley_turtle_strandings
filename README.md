## Source code for article titled: Modelling drift of cold-stunned Kemp’s ridley turtles stranding on the Dutch coast

Darshika's project on studying drift of cold-stunned Kemp's Ridley Turtles stranding on the Dutch coast. The code used for this research available in this github repository and the outputs generated at available in the public data repository at: 

### Shelf Environment:
Currents, winds, Stokes drift and temperature mean of the Northwest Shelf for (nearest) Decembers of years when turtle stranding occurred were extracted by:
```src/processing/ExtractEUWesternShelfWindData.ipynb```

and plotted using:
```src/visualizations/ShelfEnvironment.ipynb```

### Simulations:
2D Backward simulations of 10,000 particles at the nearest coastal cell from the starnding location of the Kemp's ridley turltes were run using Parcels, a Lagrangian Particle Tracking Framework. The code for the simulation is available in this repository at:
```src/simulations/RidleyBacktrackingwithWind.py```

Animations of the simulations were obtained using the following code:
```src/visualizations/SimulationAnimation.py```

### Analysis:
Cold stunning events for each temperature threshold were extracted via:
```src/analysis/ExtractColdStunningEvent.py```
This code also generates figures (see Figure 2 and Figure 3 in manuscript) to plot the time and location before stranding where the threshold temperatures were crossed.

Plots for different windage settings (Figure 4 of the manuscript) are produced using output of the previous python code and running:
```src/visualizations/WindageAnalysis.py```

Similar Windage analysis plots (see supplementary material) for per station are obtained from: 
```src/visualizations/WindagePerStation.py```






