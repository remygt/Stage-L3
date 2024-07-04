document.addEventListener('DOMContentLoaded', function() {
    const canvas = new fabric.Canvas('canvas');
    let rect;
    let isDrawing = false;
    let startPoint;

    // Charger l'image sur le canvas
    fabric.Image.fromURL('{{ image_url }}', function(img) {
        img.scaleToWidth(canvas.width);
        canvas.add(img);
        canvas.renderAll();
    });

    // Gérer le début du dessin du rectangle
    canvas.on('mouse:down', function(options) {
        isDrawing = true;
        startPoint = canvas.getPointer(options.e);

        rect = new fabric.Rect({
            left: startPoint.x,
            top: startPoint.y,
            width: 0,
            height: 0,
            fill: 'rgba(255, 0, 0, 0.3)'
        });
        canvas.add(rect);
    });

    // Mettre à jour la taille du rectangle en fonction du déplacement de la souris
    canvas.on('mouse:move', function(options) {
        if (!isDrawing) return;
        const endPoint = canvas.getPointer(options.e);
        rect.set({ width: endPoint.x - startPoint.x, height: endPoint.y - startPoint.y });
        canvas.renderAll();
    });

    // Finaliser le dessin du rectangle
    canvas.on('mouse:up', function() {
        isDrawing = false;
        if (rect.width === 0 || rect.height === 0) {
            canvas.remove(rect);
        }
    });
});

// Fonction pour enregistrer les zones dessinées
function saveZones() {
    const canvas = document.getElementById('canvas');
    const rects = canvas.getObjects('rect');
    const zonesData = [];

    rects.forEach(rect => {
        zonesData.push({
            x: rect.left,
            y: rect.top,
            width: rect.width,
            height: rect.height
            // Ajoutez le label associé à chaque zone si nécessaire
        });
    });

    // Envoi des données à votre application Flask via une requête AJAX (utilisez fetch() ou autre méthode)
    // Exemple :
    fetch('/save_zones', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(zonesData)
    })
    .then(response => {
        if (response.ok) {
            console.log('Zones enregistrées avec succès.');
        } else {
            console.error('Erreur lors de l\'enregistrement des zones.');
        }
    })
    .catch(error => {
        console.error('Erreur lors de l\'enregistrement des zones :', error);
    });
}
