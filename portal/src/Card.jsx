import { useState } from "react";
import clickMeImg from "./assets/click-me.jpg";

function Card() {
  const [imageSrc, setImageSrc] = useState(clickMeImg);
  const [file, setFile] = useState(null);
  const [report, setReport] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files && event.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile); // store file for upload

    const reader = new FileReader();
    reader.onload = () => {
      setImageSrc(reader.result); // update preview
    };
    reader.readAsDataURL(selectedFile);
  };

  const sendImage = async () => {
    if (!file) {
      alert("Please select an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:5000/process-car", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Upload failed");
      }

      const data = await res.json();
      setReport(data); // save backend response
    } catch (err) {
      console.error(err);
      alert("Error sending image to backend");
    }
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
      <button className="card-submit-button" onClick={sendImage}>
        Analyze
      </button>

      {report && (
        <div className="mt-4 p-2 border rounded bg-gray-100">
          <h3 className="font-semibold">Car Report</h3>
          <pre>{JSON.stringify(report, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default Card;
