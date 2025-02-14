import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from regions import Regions

class FITSPlotter:
    def __init__(self, image_fits, contour_fits=None, region_file=None, sigma=3e-3):
        self.image_fits = image_fits
        self.contour_fits = contour_fits
        self.region_file = region_file
        self.sigma = sigma
        
        self.hdul_base = fits.open(self.image_fits)
        self.data_base = self.hdul_base[0].data[0, 0, :, :]
        self.wcs_base = WCS(self.hdul_base[0].header).celestial

        if self.contour_fits:
            self.hdul_contour = fits.open(self.contour_fits)
            self.data_contour = self.hdul_contour[0].data[0, 0, :, :]
            self.wcs_contour = WCS(self.hdul_contour[0].header).celestial
        else:
            self.data_contour = None

        # Procesar región si se proporciona un archivo CRTF
        if self.region_file:
            self._apply_region_mask()

    def _apply_region_mask(self):
        """Recorta la imagen FITS según la región definida en el archivo CRTF."""
        try:
            regions = Regions.read(self.region_file, format="crtf")
        except Exception as e:
            print(f"Error al leer el archivo CRTF: {e}")
            return
        
        x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf

        for region in regions:
            pixel_region = region.to_pixel(self.wcs_base)
            bbox = pixel_region.bounding_box
            x_min, x_max = min(x_min, bbox.ixmin), max(x_max, bbox.ixmax)
            y_min, y_max = min(y_min, bbox.iymin), max(y_max, bbox.iymax)
        
        # Asegurar que los valores están dentro de los límites de la imagen
        y_max = min(y_max, self.data_base.shape[0])
        x_max = min(x_max, self.data_base.shape[1])
        y_min = max(y_min, 0)
        x_min = max(x_min, 0)
        
        if x_max <= x_min or y_max <= y_min:
            print("Error: La región CRTF está fuera de los límites del FITS.")
            return
        
        # Recortar la imagen base
        self.data_base = self.data_base[y_min:y_max, x_min:x_max]
        self.wcs_base = self.wcs_base.slice((slice(y_min, y_max), slice(x_min, x_max)))
        
        # Recortar la imagen de contorno si existe
        if self.data_contour is not None:
            self.data_contour = self.data_contour[y_min:y_max, x_min:x_max]

    def plot(self, contour_levels=None, save_as=None):
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': self.wcs_base})
        im = ax.imshow(self.data_base, cmap='plasma', origin='lower')

        if self.data_contour is not None and contour_levels:
            contour_levels = np.array(contour_levels) * self.sigma
            ax.contour(self.data_contour, levels=contour_levels, colors='white', linewidths=0.8,
                       transform=ax.get_transform(self.wcs_base))

        plt.colorbar(im, ax=ax, pad=0.05, label='Intensidad (Jy/beam)')
        ax.set_xlabel('Ascensión Recta (RA)')
        ax.set_ylabel('Declinación (Dec)')
        plt.title(f'Imagen con contornos de {self.contour_fits if self.contour_fits else "ninguno"}')

        if save_as:
            plt.savefig(save_as, dpi=300, bbox_inches='tight')
            print(f"Imagen guardada como {save_as}")
        
        plt.show()

    def close(self):
        self.hdul_base.close()
        if self.contour_fits:
            self.hdul_contour.close()
        print("Archivos FITS cerrados.")