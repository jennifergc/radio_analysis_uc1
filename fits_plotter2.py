import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib as mpl
from reproject import reproject_interp

class FITSPlotter:
    def __init__(self, image_fits, contour_fits=None, sigma=3e-3):
        """
        Inicializa la clase cargando los archivos FITS.

        Parámetros:
            image_fits (str): Ruta al archivo FITS de la imagen base.
            contour_fits (str, opcional): Ruta al archivo FITS de los contornos.
            sigma (float, opcional): Factor de escala para los contornos.
        """
        self.image_fits = image_fits
        self.contour_fits = contour_fits
        self.sigma = sigma

        # Cargar la imagen base
        self.hdul_base = fits.open(self.image_fits)
        self.data_base = self.hdul_base[0].data.squeeze()  # Asegurar que sea 2D
        self.wcs_base = WCS(self.hdul_base[0].header).celestial

        # Cargar imagen de contornos si está disponible
        if self.contour_fits:
            self.hdul_contour = fits.open(self.contour_fits)
            self.data_contour = self.hdul_contour[0].data.squeeze()  # Asegurar que sea 2D
            self.wcs_contour = WCS(self.hdul_contour[0].header).celestial

            # Reproyectar imagen de contornos si es necesario
            if self.data_contour.shape != self.data_base.shape:
                print("⚠️ Reproyectando contornos para que coincidan con la imagen base...")
                self.data_contour, _ = reproject_interp((self.data_contour, self.hdul_contour[0].header), 
                                                         self.hdul_base[0].header, shape_out=self.data_base.shape)
        else:
            self.data_contour = None

    def plot(self, contour_levels=None, save_as=None, title="Mapa de Intensidad", object_name="M17"):
        """
        Genera la visualización de la imagen FITS y contornos.
        """
        # Configuración de estilo
        mpl.rcParams.update({
            "font.family": "serif",
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.dpi": 300
        })

        # Crear la figura
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': self.wcs_base})
        im = ax.imshow(self.data_base, cmap='inferno', origin='lower', interpolation='nearest')

        # Si hay contornos, procesarlos
        if self.data_contour is not None:
            # 📌 DIAGNÓSTICO: Verificar los valores de la imagen de contornos antes de graficar
            print("📌 Dimensiones de self.data_contour:", self.data_contour.shape)
            print("📌 Valores únicos en self.data_contour:", np.unique(self.data_contour))
            print("📌 Min:", np.nanmin(self.data_contour), "Max:", np.nanmax(self.data_contour))
            print("📌 WCS de la imagen base:", self.wcs_base)
            print("📌 WCS de la imagen de contornos:", self.wcs_contour)
            
            # Reemplazar valores 0 con NaN para evitar errores
            self.data_contour[self.data_contour == 0] = np.nan
            
            # Ajustar niveles de contorno si no se especifican
            if contour_levels is None:
                min_val, max_val = np.nanmin(self.data_contour), np.nanmax(self.data_contour)
                contour_levels = np.linspace(min_val, max_val, num=5)
        
            # 📌 VERIFICACIÓN FINAL: Solo graficar si hay datos válidos
            if np.nanmax(self.data_contour) > 0:
                ax.contour(self.data_contour, levels=contour_levels, colors='white', linewidths=1,
                           transform=ax.get_transform(self.wcs_base))
            else:
                print("⚠️ No se pueden dibujar contornos: La imagen de contornos está vacía o fuera de escala.")


       
        
        # Configuración de ejes
        ax.set_xlabel('Ascensión Recta (RA)')
        ax.set_ylabel('Declinación (Dec)')
        ax.tick_params(axis="both", direction="in", which="both", length=6, width=1.5)

        # Barra de color
        cbar = plt.colorbar(im, ax=ax, pad=0.05)
        cbar.set_label('Intensidad (Jy/beam)')

        # Agregar título y anotaciones
        ax.text(0.05, 0.95, f"{object_name}", transform=ax.transAxes, fontsize=16, fontweight='bold',
                color='white', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        plt.title(title)

        # Guardar imagen si es necesario
        if save_as:
            plt.savefig(save_as, dpi=300, bbox_inches='tight')
            print(f"Imagen guardada como {save_as}")

        plt.show()

    def close(self):
        """Cierra los archivos FITS abiertos."""
        self.hdul_base.close()
        if self.contour_fits:
            self.hdul_contour.close()
        print("Archivos FITS cerrados.")
