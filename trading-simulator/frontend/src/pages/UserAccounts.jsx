import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { useNavigate } from "react-router-dom";


const UserAccounts = () => {
  const location = useLocation();

  const [formData, setFormData] = useState({
    name: "",
    password: "",
    apiKey: "",
    apiSecret: "",
    strategies: []
  });
  const [strategies, setStrategies] = useState([]);
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("register");
  const [editingUser, setEditingUser] = useState(null);

  // 1️⃣ Set active tab from query parameter
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const tab = params.get("tab");
    if (tab) setActiveTab(tab); // open specified tab if passed in URL
  }, [location]);

  // Fetch strategies
  useEffect(() => {
    const fetchStrategies = async () => {
      setLoading(true);
      try {
        const res = await fetch("http://localhost:3001/api/strategies/names");
        const data = await res.json();
        setStrategies(data.data || []);
      } catch (err) {
        setError(err.message);
      } finally { setLoading(false); }
    };
    fetchStrategies();
  }, []);

  // Fetch users
  useEffect(() => {
    if (activeTab === "view") fetchUsers();
  }, [activeTab]);

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:3001/api/users");
      const data = await res.json();
      setUsers(data.data || []);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const toggleStrategy = (strategyName) => {
    setFormData(prev => {
      const selected = prev.strategies.includes(strategyName);
      return {
        ...prev,
        strategies: selected
          ? prev.strategies.filter(s => s !== strategyName)
          : [...prev.strategies, strategyName]
      };
    });
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const payload = { ...formData, strategies: formData.strategies.filter(Boolean) };

    try {
      const url = editingUser
        ? `http://localhost:3001/api/users/${editingUser.id}`
        : "http://localhost:3001/api/users/register";
      const method = editingUser ? "PUT" : "POST";

      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to save user");

      alert("User saved successfully!");
      setFormData({ name: "", password: "", apiKey: "", apiSecret: "", strategies: [] });
      setEditingUser(null);
      fetchUsers();
      setActiveTab("view");
    } catch (err) { alert("Error: " + err.message); }
  };

  const handleEditUser = (user) => {
    setEditingUser(user);
    setFormData({
      name: user.name,
      password: "",
      apiKey: user.api_key || "",
      apiSecret: user.api_secret || "",
      strategies: user.strategies || []
    });
    setActiveTab("register");
  };

  return (
    <div className="min-h-screen flex flex-col items-center bg-gray-100 p-8">
      <div className="bg-white rounded-xl shadow-lg p-6 max-w-6xl w-full">
        <h1 className="text-3xl font-bold text-emerald-600 mb-6 text-center">User Accounts</h1>

        {/* Tabs */}
        <div className="flex mb-6 justify-center gap-4">
          <button onClick={() => setActiveTab("register")} className={`px-6 py-2 rounded-full font-semibold ${activeTab==="register"?"bg-emerald-600 text-white":"bg-gray-200 text-gray-700"}`}>Register</button>
          <button onClick={() => setActiveTab("view")} className={`px-6 py-2 rounded-full font-semibold ${activeTab==="view"?"bg-emerald-600 text-white":"bg-gray-200 text-gray-700"}`}>View Users</button>
        </div>

        {/* REGISTER FORM */}
        {activeTab==="register" && (
          <div className="flex flex-col md:flex-row gap-10">
            <div className="md:w-1/3">
              <h2 className="text-xl font-semibold mb-4 text-gray-800">Select Preferred Trading Strategies</h2>
              {loading && <p>Loading strategies...</p>}
              {error && <p className="text-red-500">{error}</p>}
              {!loading && !error && (
                <div className="flex flex-wrap gap-3 max-h-96 overflow-y-auto border p-4 rounded-lg bg-gray-50">
                  {strategies.map(name => (
                    <button
                      key={name}
                      type="button"
                      onClick={() => toggleStrategy(name)}
                      className={`flex items-center gap-2 px-4 py-2 rounded-full border cursor-pointer ${formData.strategies.includes(name) ? "bg-emerald-500 text-white border-emerald-600 shadow-md" : "bg-white text-gray-700 border-gray-300 hover:bg-emerald-100"}`}
                    >
                      {formData.strategies.includes(name) && <span>✔</span>}
                      {name}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <form onSubmit={handleSubmit} className="md:w-2/3 space-y-6">
              <input type="text" name="name" placeholder="Name" value={formData.name} onChange={handleChange} required className="w-full p-3 border rounded-lg"/>
              <input type="password" name="password" placeholder="Password" value={formData.password} onChange={handleChange} required className="w-full p-3 border rounded-lg"/>
              <input type="text" name="apiKey" placeholder="API Key" value={formData.apiKey} onChange={handleChange} className="w-full p-3 border rounded-lg"/>
              <input type="password" name="apiSecret" placeholder="API Secret" value={formData.apiSecret} onChange={handleChange} className="w-full p-3 border rounded-lg"/>
              <button type="submit" className="w-full bg-emerald-600 text-white py-3 rounded-lg font-semibold hover:bg-emerald-700">{editingUser?"Update User":"Register Account"}</button>
            </form>
          </div>
        )}

        {/* VIEW USERS */}
        {activeTab==="view" && (
          <div>
            {loading && <p>Loading users...</p>}
            {error && <p className="text-red-500">{error}</p>}
            {!loading && !error && users.length>0 && (
              <table className="w-full border-collapse border border-gray-300">
                <thead>
                  <tr className="bg-gray-200"><th className="border p-2">Name</th><th className="border p-2">Strategies</th><th className="border p-2">Actions</th></tr>
                </thead>
                <tbody>
                  {users.map(user => (
                    <tr key={user.id}>
                      <td className="border p-2">{user.name}</td>
                      <td className="border p-2">
                      <div className="flex flex-wrap gap-2">
                        {user.strategies.map((strategy, index) => (
                          <span
                            key={index}
                            className="bg-emerald-100 text-emerald-800 px-3 py-1 rounded-full text-sm font-medium"
                          >
                            {strategy}
                          </span>
                        ))}
                      </div>
                    </td>
                      <td className="border p-2"><button onClick={()=>handleEditUser(user)} className="bg-emerald-500 text-white px-4 py-1 rounded hover:bg-emerald-600">Edit</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {!loading && !error && users.length===0 && <p>No users found</p>}
          </div>
        )}
      </div>
    </div>
  );
};

export default UserAccounts;
