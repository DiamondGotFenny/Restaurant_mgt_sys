const getMicrophone = () => {
  return navigator.mediaDevices.getUserMedia({ audio: true, video: false });
};
export default getMicrophone;
