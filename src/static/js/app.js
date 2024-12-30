import React, { useState } from 'react';
import { Upload } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

const ImageAnalyzer = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [uploadedImages, setUploadedImages] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [showAttention, setShowAttention] = useState(false);
  const [showSegmentation, setShowSegmentation] = useState(false);

  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files);
    setUploadedImages(prevImages => [...prevImages, ...files.map(file => ({
      name: file.name,
      url: URL.createObjectURL(file)
    }))]);
  };

  const handleFolderUpload = (event) => {
    const files = Array.from(event.target.files);
    setUploadedImages(prevImages => [...prevImages, ...files.map(file => ({
      name: file.name,
      url: URL.createObjectURL(file)
    }))]);
  };

  const handleImageSelect = (image) => {
    setSelectedImage(image);
    // Simuler une requête API
    setPrediction({
      class: "Example Class",
      probability: 0.95
    });
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Barre latérale gauche - Liste d'images */}
      <div className="w-64 bg-white p-4 border-r">
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="block w-full cursor-pointer">
              <div className="flex items-center justify-center p-4 border-2 border-dashed rounded-lg hover:bg-gray-50">
                <Upload className="h-6 w-6 mr-2" />
                <span>Upload Image</span>
              </div>
              <input
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={handleFileUpload}
              />
            </label>
            <label className="block w-full cursor-pointer">
              <div className="flex items-center justify-center p-4 border-2 border-dashed rounded-lg hover:bg-gray-50">
                <Upload className="h-6 w-6 mr-2" />
                <span>Upload Folder</span>
              </div>
              <input
                type="file"
                directory=""
                webkitdirectory=""
                className="hidden"
                onChange={handleFolderUpload}
              />
            </label>
          </div>
          <div className="space-y-2">
            {uploadedImages.map((image, index) => (
              <div
                key={index}
                onClick={() => handleImageSelect(image)}
                className={`p-2 rounded cursor-pointer hover:bg-gray-100 ${
                  selectedImage === image ? 'bg-blue-100' : ''
                }`}
              >
                {image.name}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Zone centrale - Affichage de l'image */}
      <div className="flex-1 p-4">
        {selectedImage ? (
          <div className="h-full flex items-center justify-center">
            <img
              src={selectedImage.url}
              alt={selectedImage.name}
              className="max-h-full max-w-full object-contain"
            />
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-gray-400">
            Sélectionnez une image
          </div>
        )}
      </div>

      {/* Barre latérale droite - Analyse */}
      <div className="w-64 bg-white p-4 border-l">
        {prediction ? (
          <div className="space-y-4">
            <div>
              <h3 className="font-medium">Classe prédite</h3>
              <p>{prediction.class}</p>
            </div>
            <div>
              <h3 className="font-medium">Probabilité</h3>
              <p>{(prediction.probability * 100).toFixed(1)}%</p>
            </div>
            <button
              onClick={() => setShowAttention(!showAttention)}
              className="w-full py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Visualiser l'attention
            </button>
            <button
              onClick={() => setShowSegmentation(!showSegmentation)}
              className="w-full py-2 bg-green-500 text-white rounded hover:bg-green-600"
            >
              Segmenter les régions
            </button>
          </div>
        ) : (
          <div className="text-gray-400 text-center">
            Aucune analyse disponible
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageAnalyzer;