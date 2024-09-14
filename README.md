# Analysis of the Blazhko Effect for Field RR Lyrae Stars using LINEAR and ZTF Light Curves

### Abstract
We analyzed the incidence and properties of stars that show evidence for amplitude, period, and phase modulation (the so-called Blazhko Effect) in a sample of ∼3,000 field RR Lyrae stars with LINEAR and ZTF light curve data. A preliminary subsample of about ∼240 stars was selected using various light curve statistics, and then ∼140 stars were confirmed visually as displaying the Blazhko effect. Although there are close to 8,000 Blazhko stars discovered in the Galactic bulge and LMC/SMC by the OGLE-III survey, only about 200 stars are reported in all the studies of field RR Lyrae stars to date.

### Methods
Analysis was done using Python, including various other modules such as Pandas, Numpy, etc. LINEAR and ZTF light curves were accessed using the astroML module and using equatorial coordinates to select ZTF pairs using ztfquery. Methods of analyzing RR Lyrae were $chi^2$ calculation, Lomb-Scargle periodogram analysis, and creating an algorithm for recognizing local peaks in periodograms of potential Blazhko stars. After analysis, using another algorithm with a score based mechanism, 239 stars were selected for visual analysis.

### Results & discussion
136 Blazhko stars were found in this sample of LINEAR and ZTF pairs of RR Lyrae, with a selection algorithm percentage of 57%. The total incidence rate is 4.67%, which aligns with other larger surveys of Blazhko stars. We found that extremely small differences in period and amplitude are present for Blazhko stars, hence a long temporal baseline as well as very precise data is needed for future Blazhko effect research. We also note that the time difference between LINEAR and ZTF (around 10 years) shows that the effect isn't present in both datasets for every star. This could mean imprecise data, but also that the Blazhko effect sometimes isn't detectable in stars. 

---
The notebooks in this repository are numbered for easy reading, so readers can fully understand the research process. 
