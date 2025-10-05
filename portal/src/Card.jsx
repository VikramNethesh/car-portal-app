import { useState } from "react";
import clickMeImg from "./assets/click-me.jpg";

function Card() {
  const [imageSrc, setImageSrc] = useState(clickMeImg);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      setImageSrc(reader.result); // update state with uploaded image
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="card">
      <h1 className="Heading">Vehicle Feature Extractor</h1>
      <label
        htmlFor="imageUpload"
        className="card-upload-button"
        style={{ backgroundImage: `url(${imageSrc})` }}
      ></label>

      <input
        type="file"
        id="imageUpload"
        accept=".png, .jpg, .jpeg"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />

      <br />
      <button className="card-submit-button">Analyze</button>
    </div>
  );
}

export default Card;
